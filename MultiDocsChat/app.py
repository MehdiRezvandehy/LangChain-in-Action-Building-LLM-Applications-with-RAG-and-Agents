import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
import streamlit as st

import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader # to load pdf files
from langchain.document_loaders import Docx2txtLoader # to load word files
from langchain.document_loaders import TextLoader # to load text files
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import streamlit as st

# this library makes our life easier when it 
# comes to chatting within a library
from langchain.chains.question_answering import load_qa_chain 
from streamlit_chat import message # pip install streamlit_chat

load_dotenv(find_dotenv())

# read OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = "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"

openai.api_key = os.getenv("OPENAI_API_KEY") 


#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

#### === packages to install ====
# pip install langchain pypdf openai chromadb tiktoken docx2txt

#-------- Streamlit front-end #--------
st.title("QA Bot for Documents by Langchain")
st.header("You can ask anything about your document... 🤖")

# load a pdf file

files = st.file_uploader("Please upload your files", accept_multiple_files=True,
                             type=["txt", "docx", "pdf"])

if files:
   documents = []
   if files is not None:
       for ifiles in files:
           if ifiles.name[-4:] == '.txt':
               loader = TextLoader(ifiles.name)
               documents.extend(loader.load())
           elif ifiles.name[-5:] == '.docx' or ifiles.name[-4:] == '.doc':
               loader = Docx2txtLoader(ifiles.name)
               documents.extend(loader.load())            
           elif ifiles.name[-4:] == '.pdf':
               loader = PyPDFLoader(ifiles.name)
               documents.extend(loader.load())
   
   # load files
   chat_history = []
   
   # split the data into chunks
   text_splitter = CharacterTextSplitter(
       chunk_size=500,
       chunk_overlap=5
   )
   docs = text_splitter.split_documents(documents)
   
   # create vector db chromadb
   vectordb = Chroma.from_documents(
       documents=documents,
       embedding=OpenAIEmbeddings(),
       persist_directory='./data'
   )
   vectordb.persist()
   
   chain_qa = ConversationalRetrievalChain.from_llm(
       llm,
       vectordb.as_retriever(search_kwargs={'k': 5}),
       return_source_documents=True,
       verbose=False
   )
   
   
   if 'produced' not in st.session_state:
       st.session_state['produced'] = []
       
   if 'old' not in st.session_state:
       st.session_state['old'] = []
   
   
   # get the user input
   user_input = st.chat_input("Ask a question from your documents...")
   if user_input:
       result = chain_qa({'question': user_input, 'chat_history': chat_history})
       st.session_state.old.append(user_input)
       st.session_state.produced.append(result['answer'])
       
       
   if st.session_state['produced']:
       for i in range(len(st.session_state['produced'])):
           message(st.session_state['old'][i], is_user=True, key=str(i)+ '_user')
           message(st.session_state['produced'][i], key=str(i))
