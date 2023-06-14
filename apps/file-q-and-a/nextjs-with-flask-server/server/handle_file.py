import logging
import sys
import os
import hashlib

from numpy import array, average
from config import *

from langchain import OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredWordDocumentLoader


from utils import get_embeddings, get_pinecone_id_for_file_chunk

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Handle a file by extracting its text, creating embeddings, and upserting them to Pinecone
def handle_file(file, session_id, pinecone_index, tokenizer):
    """Handle a file by extracting its text, creating embeddings, and upserting them to Pinecone."""
    filename = file.filename
    logging.info("[handle_file] Handling file: {}".format(filename))
    # # Get the file text dict from the current app config
    # file_text_dict = current_app.config["file_text_dict"]

    # save file string to fileDic
    fileDic = f"/home/ec2-user/tmpfile"
    if not os.path.exists(fileDic):
        os.makedirs(fileDic)
    filePath = os.path.join(fileDic, filename)
    file.save(filePath)

    # loader
    try:
        if file.mimetype == "application/pdf":
            loader = UnstructuredPDFLoader(filePath, mode="elements")
        elif file.mimetype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = UnstructuredWordDocumentLoader(filePath, mode="elements")
        else:
            # Unsupported file type
            raise ValueError("Unsupported file type: {}".format(file.mimetype))
    except ValueError as e:
        logging.error("[handle_file] Error extracting text from file: {}".format(e))
        raise e
    docs = loader.load()
   
    # splitters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        add_start_index=True,
    )
    docsChunks = text_splitter.split_documents(docs)
    logging.warning(len(docsChunks))

    # vectorstone
    i = 1
    vectorIds = []
    for i, doc in enumerate(docsChunks):
        vectorIds.append(get_pinecone_id_for_file_chunk(session_id, filename, i))
    docsearch= Pinecone.from_documents(documents=docs, text_key="text",ids=vectorIds, embedding=OpenAIEmbeddings(),index_name=PINECONE_INDEX, namespace=session_id)
    # simDocs = docsearch.similarity_search("summary")
   
    # summary (todo: use llamaIndex summary)
    output_summary = "True"
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
    return output_summary
