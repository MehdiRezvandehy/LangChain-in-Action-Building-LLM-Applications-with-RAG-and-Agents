import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd
import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader # to load pdf files
from langchain.chains import RetrievalQA
from langchain.document_loaders import Docx2txtLoader # to load word files
from langchain.document_loaders import TextLoader # to load text files
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from langchain.prompts import PromptTemplate
import time

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
st.title("Document QA Bot powered by LangChain")
st.header("Feel free to ask any questions about your document... 🤖")


# Load CSV files
files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=["csv"])

if files:
    dfs = []
    clmns = ['ArticleTitle', 'Question', 'Answer']
    
    # Read each uploaded CSV file and filter the required columns
    for file in files:
        df = pd.read_csv(file)
        if all(col in df.columns for col in clmns):
            dfs.append(df[clmns])
        else:
            st.warning(f"File {file.name} does not contain the required columns: {clmns}")

    # Concatenate all DataFrames
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
    else:
        st.warning("No valid files uploaded.")

    
    # Example function to simulate a time-consuming task
    def long_task():
        time.sleep(3)  # Simulate a 5-second task

    
    # Display spinner while running the long task
    with st.spinner("Please wait, processing..."):
        long_task()

    merged_df.reset_index(drop=True, inplace=True)
    merged_df = merged_df.drop_duplicates(
    subset=['ArticleTitle', 
            'Question'], 
    keep='first').reset_index(drop=True)
    
    st.write(merged_df.head())

    questions_answers = []
    for index, row in merged_df.iterrows():
        txt = f"ArticleTitle: {row['ArticleTitle']}, Question: {row['Question']}, Answer: {row['Answer']}"
        questions_answers.append(txt+"\n")
    questions_answers = ' '.join(questions_answers)

    # Split the data into chunks
    text_splitter = CharacterTextSplitter(separator="\n",
        chunk_size=800,
        chunk_overlap=400
    )
    qa_chunks = text_splitter.split_text(questions_answers)

    prompt_template = """
    {question}
    """

    # Define the PromptTemplate with the custom template
    prompt = PromptTemplate(
        input_variables=["question"],  # The variable used inside the template
        template=prompt_template  # The custom template defined above
    )

    embedding = OpenAIEmbeddings()
    
    # create our vector db chromadb
    vectordb = Chroma.from_texts(
        texts=qa_chunks,
        embedding=embedding,
        persist_directory='./chroma_qa'
    )
    vectordb.persist()
    
    # Display spinner while running the long task
    with st.spinner("Please wait, processing..."):
        long_task()

    # RetrievalQA chain to get info from the vectorstore
    chain_qa = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True,
        verbose=True
    )
    # load files
    chat_history = []

    if 'produced' not in st.session_state:
        st.session_state['produced'] = []
        
    if 'old' not in st.session_state:
        st.session_state['old'] = []

    # get the user input
    user_input = st.chat_input("Ask a question from your documents...")
    formatted_query = prompt.format(question=user_input)
    if user_input:
        result = chain_qa({'query': formatted_query, 'chat_history': chat_history})
        st.session_state.old.append(formatted_query)
        st.session_state.produced.append(result['result'])
        
        
    if st.session_state['produced']:
        for i in range(len(st.session_state['produced'])):
            message(st.session_state['old'][i], is_user=True, key=str(i)+ '_user')
            message(st.session_state['produced'][i], key=str(i))

    # Display spinner while running the long task
    with st.spinner("Please wait, processing..."):
        long_task()