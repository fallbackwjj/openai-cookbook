import logging
import sys
import os
import json
import openai
import boto3

from numpy import array, average
from config import *

from langchain import OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Pinecone
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI, OpenAIChat
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from langchain.document_loaders import S3FileLoader

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
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.indices.document_summary import DocumentSummaryIndexRetriever

from common.utils import calculate_md5, get_pinecone_id_for_file_chunk

# import nest_asyncio
# nest_asyncio.apply()


# Set up logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("debug.log"),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Handle a file by extracting its text, creating embeddings, and upserting them to Pinecone
def handle_file(file, pinecone_index):
    """Handle a file by extracting its text, creating embeddings, and upserting them to Pinecone."""
    output_summary = "True"
    filename = file.filename
    fileDic = f"/home/ec2-user/tmpfile"
    filePath = os.path.join(fileDic, filename)
    fileMd5 = calculate_md5(filePath) # as vectorstore uniq 
    logging.info(fileMd5)
    # loader
    try:
        if file.content_type == "application/pdf":
            loader = UnstructuredPDFLoader(filePath, rode="elements")
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = UnstructuredWordDocumentLoader(filePath, mode="elements")
        else:
            # Unsupported file type
            raise ValueError("Unsupported file type: {}".format(file.content_type))
    except ValueError as e:
        logging.error("[handle_file] Error extracting text from file: {}".format(e))
        raise e
    docsChunks = loader.load_and_split()
    # todo use redis as files exist or noexist
    docsChunks.append(Document(page_content="ping", metadata=dict())) 

    # vectorstone
    embedding=OpenAIEmbeddings()
    # embedding=OpenAIEmbeddings(
    #     deployment="demo", 
    #     openai_api_base="https://preonline.openai.azure.com/"",
    #     openai_api_type="azure",
    # )  
    # vectorstore = Pinecone(pinecone_index, embedding.embed_query, "text")
    docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX, embedding=embedding,
        text_key="text", namespace=PINECONE_FILE_NAMESPACE)
    pingRes = docsearch.similarity_search(query="ping", filter={"md5":fileMd5}, k = 1)
    logging.warning(f"pingRes:{pingRes}")
    if pingRes : 
        return {"summary" : output_summary, "md5" : fileMd5}

    # splitters
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=800,
    #     chunk_overlap=80,
    #     length_function=len,
    #     add_start_index=True,
    # )
    # docsChunks = text_splitter.create_documents(docs)

    # vectorstone
    vectorIds = []
    for i, doc in enumerate(docsChunks):
        vectorIds.append(str(fileMd5+"-!"+str(i)))
        doc.metadata["md5"] = fileMd5 # for md5 filter in answer_queation.py
    logging.info(docsChunks)
    docsearch= Pinecone.from_documents(documents=docsChunks, embedding=embedding, 
        index_name=PINECONE_INDEX, namespace=PINECONE_FILE_NAMESPACE,
        text_key="text", ids=vectorIds)
    

    '''
    # summary (todo: use llamaIndex summary)
    # llm = AzureOpenAI(
    #     deployment_name="demo",
    # )
    # response = llm("Tell me a joke")
    # os.environ["OPENAI_API_TYPE"] = "azure"
    # os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
    # os.environ["OPENAI_API_BASE"] = "https://preonline.openai.azure.com/"
    # os.environ["OPENAI_API_KEY"] = "2403924da44040f9aa9c744443051707"
    
    # openai.api_type = "azure"
    # openai.api_base = "https://preonline.openai.azure.com/"
    # openai.api_version = "2023-03-15-preview"
    # openai.api_key = "2403924da44040f9aa9c744443051707"

    documents = SimpleDirectoryReader(input_files=[filePath]).load_data()
    logging.info(len(documents))
    # llm = AzureOpenAI(
    #     deployment_name="demo", 
    #     openai_api_type=openai.api_type,
    #     openai_api_version=openai.api_version,
    #     openai_api_key=openai.api_key,
    #     openai_api_base=openai.api_base,
    # )
    llm=ChatOpenAI(
        model=GENERATIVE_MODEL,  
        max_tokens=500,
    )
    llm_predictor = LLMPredictor(llm=llm)
    # embedding_llm = LangchainEmbedding(
    #     OpenAIEmbeddings(
    #         deployment="demo",
    #         model=EMBEDDINGS_MODEL,
    #         openai_api_key=openai.api_key,
    #         openai_api_base=openai.api_base,
    #         openai_api_type=openai.api_type,
    #         openai_api_version=openai.api_version,
    #     ),
    #     embed_batch_size=1,
    # )
  
    summary_query = (
        "Give a concise summary of this document in bullet points. Also describe some of the questions "
        "that this document can answer."
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        callback_manager=CallbackManager([LlamaDebugHandler(print_trace_on_end=True)]),
        # embed_model=embedding_llm,
    )
    response_synthesizer = ResponseSynthesizer.from_args(
        response_mode="tree_summarize", 
        use_async=True,
        service_context=service_context,
    )
    doc_summary_index = DocumentSummaryIndex.from_documents(
        documents, 
        service_context=service_context,
        response_synthesizer=response_synthesizer,
        summary_query=summary_query,
    )
    # query_engine = doc_summary_index.as_query_engine(
    #     response_mode='tree_summarize',
    #     verbose=True,
    # )
    # response =  query_engine.query(summary_query)
    # logging.warning(response)
    # logging.warning(llm_predictor.last_token_usage)
    '''

    '''
    prompt_template = """Generate a summary of the following document, arranging the top 5 points in sequential order starting from 1, with the highest semantic relevance, while maintaining the language consistency of the document: 
    
    {text}
    
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(ChatOpenAI(temperature=0, model=GENERATIVE_MODEL, max_tokens=500), 
        chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT) 
    # chain = load_summarize_chain(OpenAI(temperature=0, model="curie", max_tokens=300), 
    #     chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT, ver)
    output_summary = chain({"input_documents": docs}, return_only_outputs=True)  
    logging.warning(output_summary)      
    
    wrapped_text = textwrap.fill(output_summary, 
        width=100,
        break_long_words=False,
        replace_whitespace=False)

    refine_template = ("Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary"
        "If the context isn't useful, return the original summary."
        "Write a concise bullet point summary."
    )
    question_prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    refine_prompt = PromptTemplate(input_variables=["existing_answer", "text"], template=refine_template)
    chain = load_summarize_chain(OpenAI(temperature=0, model=GENERATIVE_MODEL, max_tokens=1000), chain_type="refine", 
        return_intermediate_steps=True, question_prompt=question_prompt, refine_prompt=refine_prompt)
    output_summary = chain({"input_documents": docs}, return_only_outputs=True)
    '''
    return {"summary" : output_summary, "md5" : fileMd5}


# upload file
def upload_file(file, pinecone_index):
    """Handle a file by extracting its text, creating embeddings, and upserting them to Pinecone."""
    output_summary = "True"
    filename = file.filename
    fileDic = f"/home/ec2-user/tmpfile"
    filePath = os.path.join(fileDic, filename)
    fileMd5 = calculate_md5(filePath) # as vectorstore uniq 
    logging.info(fileMd5)

    # vectorstone
    embedding=OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX,
        embedding=embedding,
        text_key="text", 
        namespace=PINECONE_FILE_NAMESPACE,
    )
    pingRes = docsearch.similarity_search(query="ping", filter={"md5":fileMd5}, k = 1)
    logging.warning(f"pingRes:{pingRes}")
    if pingRes : 
        return {"summary" : output_summary, "md5" : fileMd5}

    # vectorstone
    vectorIds = []
    for i, doc in enumerate(docsChunks):
        vectorIds.append(str(fileMd5+"-!"+str(i)))
        doc.metadata["md5"] = fileMd5 # for md5 filter in answer_queation.py
    logging.info(docsChunks)
    docsearch= Pinecone.from_documents(documents=docsChunks, embedding=embedding, 
        index_name=PINECONE_INDEX, namespace=PINECONE_FILE_NAMESPACE,
        text_key="text", ids=vectorIds)
    
    return {"summary" : output_summary, "md5" : fileMd5}