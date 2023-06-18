import logging
import sys
import os
import json
import openai
import boto3
import tempfile

from numpy import array, average
from config import *

from fastapi import  Depends
from sqlalchemy.orm import Session

from langchain import OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
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
from persistence.database import get_db, get_local_db
from persistence.model.message import Message
from persistence.model.channel import Channel, ChannelResSchemas
from persistence.model.file_meta import FileMeta, FilemetaResSchemas
from persistence.repository.channel_repository import ChannelRepository
from persistence.repository.message_repository import MessageRepository
from persistence.repository.file_meta_repository import FileMetaRepository

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
async def handle_file(pinecone_index, file, fileMd5, filePath):
    """Handle a file by extracting its text, creating embeddings, and upserting them to Pinecone."""
    summay = ""
    db = get_local_db()
    try:
        fileMeta = FileMetaRepository(db).get_file_by_md5(fileMd5) 
        fileMetaJson = FilemetaResSchemas.from_orm(fileMeta)
        logging.info(fileMetaJson)
        if fileMeta.file_is_handle == 1 :
            logging.info(f"handle_file : {file.filename} had processed")
            db.close()
            return fileMetaJson.summary
        
        # loader cur: deal local file  todo: deal s3 file START
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

        # vectorstone
        vectorIds = []
        for i, doc in enumerate(docsChunks):
            vectorIds.append(str(fileMd5+"-!"+str(i)))
            doc.metadata["md5"] = fileMd5 # for md5 filter in answer_queation.py
        logging.info(docsChunks)
        docsearch= Pinecone.from_documents(
            documents=docsChunks, 
            embedding=OpenAIEmbeddings(), 
            index_name=PINECONE_INDEX, 
            namespace=PINECONE_FILE_NAMESPACE,
            text_key="text", 
            ids=vectorIds,
        )

        ### langchain summary start ###
        prompt_template = """Generate a summary of the following document, arranging the top 5 points in sequential order starting from 1, with the highest semantic relevance, while maintaining the language consistency of the document: 
        
        {text}
        
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(
            # llm = ChatOpenAI(temperature=0, model=GENERATIVE_MODEL, max_tokens=500), 
            llm = AzureChatOpenAI(
                deployment_name="demo", 
                openai_api_type="azure",
                openai_api_version="2023-03-15-preview",
                openai_api_key="2403924da44040f9aa9c744443051707",
                openai_api_base="https://preonline.openai.azure.com/",
                n = 5,
            ),
            chain_type="map_reduce", 
            map_prompt=PROMPT, 
            combine_prompt=PROMPT,
        ) 
        output_summary = chain({"input_documents": docsChunks,} , return_only_outputs=True)  
        logging.warning(output_summary) 
        summay = output_summary["output_text"]
        # wrapped_text = textwrap.fill(output_summary, 
        #     width=100,
        #     break_long_words=False,
        #     replace_whitespace=False)

        # refine_template = ("Your job is to produce a final summary\n"
        #     "We have provided an existing summary up to a certain point: {existing_answer}\n"
        #     "We have the opportunity to refine the existing summary"
        #     "(only if needed) with some more context below.\n"
        #     "------------\n"
        #     "{text}\n"
        #     "------------\n"
        #     "Given the new context, refine the original summary"
        #     "If the context isn't useful, return the original summary."
        #     "Write a concise bullet point summary."
        # )
        # question_prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
        # refine_prompt = PromptTemplate(input_variables=["existing_answer", "text"], template=refine_template)
        # chain = load_summarize_chain(OpenAI(temperature=0, model=GENERATIVE_MODEL, max_tokens=1000), chain_type="refine", 
        #     return_intermediate_steps=True, question_prompt=question_prompt, refine_prompt=refine_prompt)
        # output_summary = chain({"input_documents": docsChunks}, return_only_outputs=True)
        ### summary end ###
        fileMeta.summary = output_summary["output_text"]
        fileMeta.file_is_handle = 1
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
    return summay

def llama_summary(filePath):
    ## see https://gpt-index.readthedocs.io/en/latest/examples/index_structs/doc_summary/DocSummary.html
    documents = SimpleDirectoryReader(input_files=[filePath]).load_data()
    logging.info(len(documents))
    llm = AzureChatOpenAI(
        deployment_name="demo", 
        openai_api_type="azure",
        openai_api_version="2023-03-15-preview",
        openai_api_key="2403924da44040f9aa9c744443051707",
        openai_api_base="https://preonline.openai.azure.com/"
    )
    # llm=ChatOpenAI(
    #     model=GENERATIVE_MODEL,  
    #     max_tokens=500,
    # )
    llm_predictor = LLMPredictor(llm=llm)
    embedding_llm = LangchainEmbedding(
        OpenAIEmbeddings(
            deployment="demo",
            model=EMBEDDINGS_MODEL,
            openai_api_type="azure",
            openai_api_version="2023-03-15-preview",
            openai_api_key="2403924da44040f9aa9c744443051707",
            openai_api_base="https://preonline.openai.azure.com/"
        ),
        embed_batch_size=1,
    )
    summary_query = (
        "Give a concise summary of this document in bullet points. Also describe some of the questions "
        "that this document can answer."
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        callback_manager=CallbackManager([LlamaDebugHandler(print_trace_on_end=True)]),
        embed_model=embedding_llm,
    )
    # response_synthesizer = ResponseSynthesizer.from_args(
    #     response_mode="tree_summarize", 
    #     use_async=True,
    #     service_context=service_context,
    # )
    doc_summary_index = DocumentSummaryIndex.from_documents(
        documents, 
        service_context=service_context,
        summary_query=summary_query,
        # response_synthesizer=response_synthesizer,
    )
    # query_engine = doc_summary_index.as_query_engine(
    #     response_mode='tree_summarize',
    #     verbose=True,
    # )
    # response =  query_engine.query(summary_query)
    # logging.warning(response)
    logging.warning(llm_predictor.last_token_usage)