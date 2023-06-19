from config import *
import nltk
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader, 
    LLMPredictor,
    ServiceContext,
    ResponseSynthesizer,
    LangchainEmbedding,
    MockLLMPredictor,
    SimpleKeywordTableIndex,
    RAKEKeywordTableIndex, 
    StorageContext
)
from llama_index.retrievers import  KeywordTableSimpleRetriever
from llama_index.indices.keyword_table.retrievers import KeywordTableRAKERetriever


from llama_index.indices.document_summary import DocumentSummaryIndex,DocumentSummaryIndexRetriever

from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.readers.pinecone import PineconeReader
from llama_index.vector_stores.types import (
    ExactMatchFilter, 
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores import PineconeVectorStore
from llama_index import Document
from llama_index.query_engine import RetrieverQueryEngine
from persistence.model.message import Message, MessageResSchemas, RoleEnum
from persistence.model.channel import Channel, ChannelResSchemas




from common.utils import calculate_md5, get_pinecone_id_for_file_chunk

import nest_asyncio
nest_asyncio.apply()
nltk.download('stopwords')

def get_answer_from_files(question):
    userMsg = question.message
    systemMsg = question.sysMessage
    fileMd5 = question.md5
    # get_relevant_documents
    embedding=OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX, embedding=embedding,
        text_key="text", namespace=PINECONE_FILE_NAMESPACE)
    simDocs = docsearch.similarity_search(query=userMsg, filter={"md5":fileMd5}, k = 200)
    logging.info(f":simDocs{simDocs}")
    files_string = ""
    for i, doc in enumerate(simDocs):
        files_string += f"\n{doc.page_content}\n"
    # logging.warning(f"files_string {files_string}")
    files_string = files_string[0:1500]

    # ask chatgpt
    prompt=ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(systemMsg), 
            HumanMessagePromptTemplate.from_template(userMsg)
        ]
    )
    promptMsg=prompt.format_prompt(files_string=files_string, question=userMsg).to_string()
    chain = LLMChain(
        llm=ChatOpenAI(
            model=GENERATIVE_MODEL, 
            temperature=0, 
            max_tokens=500, 
            callbacks=[StreamingStdOutCallbackHandler()],
        ),
        prompt=prompt,
        verbose=True,
    )
    answer = chain.run(files_string=files_string, question=userMsg)
    resDict = {
        "Ask" : userMsg,
        "Answer" : answer,
        "Prompt" : promptMsg,
    }
    # add search index
    # update_chat_history(pinecone_index, fileMd5, resDict)
    return resDict

def update_chat_history(pinecone_index, q, messageList, channelList):
    messageList = [MessageResSchemas.from_orm(msg) for msg in messageList]
    channelList = [ChannelResSchemas.from_orm(msg) for msg in channelList]
    docs = [Document(msg.channel_name, extra_info= {"message_id" :"", "channel_id" : msg.channel_id}) for msg in channelList]
    docsChannel = [Document(msg.content, extra_info = {"message_id" :msg.message_id, "channel_id" : msg.channel_id}) for msg in messageList]
    docs.extend(docsChannel)
    # logging.info(f"docsinfo: {docs}")

    # service_context = ServiceContext.from_defaults(chunk_size=1024)
    # node_parser = service_context.node_parser
    # nodes = node_parser.get_nodes_from_documents(docs)

    # storage_context = StorageContext.from_defaults()
    # storage_context.docstore.add_documents(nodes)
    # doc_rake_index = RAKEKeywordTableIndex(
    #     nodes, 
    #     storage_context=storage_context,
    #     max_keywords_per_chunk = 2,
    # )
  
    # vector_store = PineconeVectorStore(
    #     pinecone_index=pinecone_index, 
    #     index_name=PINECONE_INDEX,
    #     namespace=PINECONE_CHAT_NAMESPACE,
    # )
    # storage_context = StorageContext.from_defaults(
    #     vector_store=vector_store
    # )
    service_context = ServiceContext.from_defaults(
        callback_manager=CallbackManager([LlamaDebugHandler(print_trace_on_end=True)]),
    )
    doc_rake_index = RAKEKeywordTableIndex.from_documents(
        docs,
        service_context=service_context,
        # storage_context=storage_context
    )
    doc_rake_retriever = KeywordTableRAKERetriever(index=doc_rake_index)
    response = doc_rake_retriever.retrieve(q)
    # logging.warning(response.)
   

    # keyword query engine
    # response_synthesizer = ResponseSynthesizer.from_args()
    # keyword_query_engine = RetrieverQueryEngine(
    #     retriever=doc_rake_retriever,
    #     response_synthesizer=response_synthesizer,
    # )
    # res = keyword_query_engine.query(q)
    # logging.warning(res)
    # demo = vector_store.query(
    #     query = VectorStoreQuery(
    #         mode = "default",
    #         similarity_top_k = 1000,
    #     )
    # )
    # logging.warning(demo)
    # loaded_index = VectorStoreIndex.from_vector_store(
    #     vector_store=vector_store
    # )
   
