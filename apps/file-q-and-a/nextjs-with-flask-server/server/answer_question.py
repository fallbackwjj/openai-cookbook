from config import *
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

def get_answer_from_files(question, session_id):
    # parse question
    # logging.warning(f"Getting answer for question: {question}")
    userMsg = list(filter(lambda item: item['role'] == 'user', question['messages']))
    systemMsg = list(filter(lambda item: item['role'] == 'system', question['messages']))
    last_user_msg = userMsg[-1] if userMsg else None
    last_system_msg = systemMsg[-1] if systemMsg else None
    fileMd5 = question["files"][0]["md5"]
    logging.warning(f"fileMd5 {fileMd5}")
    logging.warning(f"system: {last_system_msg}")
    logging.warning(f"user {last_user_msg}")
    
    try:
        # get_relevant_documents
        docsearch = Pinecone.from_existing_index(index_name=PINECONE_INDEX, embedding=OpenAIEmbeddings(),
            text_key="text", namespace=PINECONE_BOE_NAMESPACE)
        simDocs = docsearch.similarity_search(query=last_user_msg['content'], filter={"md5":fileMd5}, k = 200)
        logging.warning(f":simDocs{simDocs}")
        files_string = ""
        for i, doc in enumerate(simDocs):
            files_string += f"\n{doc.page_content}\n"
        logging.warning(f"files_string {files_string}")
        files_string = files_string[0:1500]
        # ask chatgpt
        prompt=ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(last_system_msg["content"]), 
                HumanMessagePromptTemplate.from_template(last_user_msg["content"])
            ]
        )
        promptMsg=prompt.format_prompt(files_string=files_string, question=last_user_msg["content"]).to_string()
        chain = LLMChain(
            llm=ChatOpenAI(model=GENERATIVE_MODEL, temperature=0, max_tokens=500),
            prompt=prompt,
            verbose=True,
        )
        answer = chain.run(files_string=files_string, question=last_user_msg["content"])
        resDict = {
            "Answer" : answer,
            "Prompt" : promptMsg,
        }
        resDictStr = '\n'.join(f'{k} ===> \n  {v}\n' for k, v in resDict.items())
        logging.warning(f"[get_answer_from_files] answer: {resDictStr}")
        return resDictStr

    except Exception as e:
        logging.warning(f"[get_answer_from_files] error: {e}")
        return str(e)

