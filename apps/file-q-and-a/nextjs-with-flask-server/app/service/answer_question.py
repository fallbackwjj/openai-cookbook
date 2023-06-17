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
    RAKEKeywordTableIndex, 
    StorageContext
)
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
from llama_index.retrievers import  KeywordTableSimpleRetriever


from common.utils import calculate_md5, get_pinecone_id_for_file_chunk

import nest_asyncio
nest_asyncio.apply()
nltk.download('stopwords')

def get_answer_from_files(question, pinecone_index):
    userMsg = question.userMessage
    systemMsg = question.sysMessage
    fileMd5 = question.channelId
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
    resDictStr = '\n'.join(f'{k} ===> \n  {v}\n' for k, v in resDict.items())
    logging.info(f"[get_answer_from_files] answer: {resDictStr}")
    
    # add search index
    update_chat_history(pinecone_index, fileMd5, resDict)
    return resDictStr

def update_chat_history(pinecone_index, fileMd5, chatHistory):
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, 
        index_name=PINECONE_INDEX,
        namespace=PINECONE_CHAT_NAMESPACE,
    )
    demo = vector_store.query(
        query = VectorStoreQuery(
            mode = "default",
            similarity_top_k = 1000,
        )
    )
    logging.warning(demo)
    # loaded_index = VectorStoreIndex.from_vector_store(
    #     vector_store=vector_store
    # )
    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(llm=ChatOpenAI(model=GENERATIVE_MODEL)),
        callback_manager=CallbackManager([LlamaDebugHandler(print_trace_on_end=True)]),
    )
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    doc_rake_index = RAKEKeywordTableIndex.from_documents(
        [
            Document(chatHistory["Ask"], extra_info={"md5" : fileMd5}),
            Document(chatHistory["Answer"], extra_info={"md5" : fileMd5}),
        ], 
        service_context=service_context,
        storage_context=storage_context
    )
    doc_rake_retriever = KeywordTableSimpleRetriever(index=doc_rake_index)
    response = doc_rake_retriever.retrieve("⻜书视频")
    logging.warning(response)