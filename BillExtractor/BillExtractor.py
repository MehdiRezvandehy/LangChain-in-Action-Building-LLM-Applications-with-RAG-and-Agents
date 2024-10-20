from langchain.llms import OpenAI
from pypdf import PdfReader
import pandas as pd
import re
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import PyPDF2
import streamlit as st
import openai
import os
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


# get Info from PDF file
def pdf_text(pdf_doc):
    text=""  # make empty text
    pdf_reader = PyPDF2.PdfReader(pdf_doc)
    for page in pdf_reader.pages:  # read each page and convert to text
        text += page.extract_text()
    return text


# read OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = "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"

openai.api_key = os.getenv("OPENAI_API_KEY") 

# get data from text of pdf
def extracted_data(pages_data):
    template = """Extract all the following values: "Previous balance", "Electricity", "Natural Gas", 
         "Water Treatment and Supply", "Wastewater Collection and Treatment", 
         "Stormwater Management", "Waste and Recycling", "Due Date" and "Total Amount Due".
         First read the text to find the key phrase .
         {pages}

        Expected output: dolloar sign should be removed
        {{"Due Date": "2024 March 05", "Total Amount Due": 4568, "Previous balance": 546, "Electricity": 124, "Natural Gas": 452, "Water Treatment and Supply": 456, "Wastewater Collection and Treatment": 145, "Stormwater Management": 12, "Waste and Recycling": 12}}
        Please notice "Due Date" comes after "If payment is received after".
        """
    prompt_template = PromptTemplate(input_variables=["pages"], template=template)
    llm = OpenAI(temperature=0.0)
    full_response = llm(prompt_template.format(pages=pages_data))
    
    full_response = full_response.replace('\n','')
    return full_response


# create documents from the uploaded pdfs
def create_docs(user_pdf_list):
    df = pd.DataFrame({"Due Date": pd.Series(dtype='str'),  
                   "Total Amount Due": pd.Series(dtype='str'),
                   "Previous balance": pd.Series(dtype='int'),
                   "Electricity": pd.Series(dtype='str'),
                   "Natural Gas": pd.Series(dtype='str'),
                   "Wastewater Collection and Treatment": pd.Series(dtype='str'),
                   "Stormwater Management": pd.Series(dtype='int'),
                   "Water Treatment and Supply": pd.Series(dtype='str'),
                   "Waste and Recycling": pd.Series(dtype='str')
                    })    

    for filename in user_pdf_list:
        
        print(filename)
        raw_data = pdf_text(filename)
        print(raw_data)
        print("extracted raw data")

        #key_phrase1 = "If payment "
        #key_phrase2 = "Free Outside Alberta:"
        llm_extracted_data = extracted_data(raw_data)

        pattern = r'{(.+)}' # capture one or more of any character, except newline
        match = re.search(pattern, llm_extracted_data, re.DOTALL)

        if match:
            extracted_text = match.group(1)
            # Converting the extracted text to a dictionary
            data_dict = eval('{' + extracted_text + '}')
            print(data_dict)
        else:
            print("Nothing found.")

     
        df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)

        print("********************DONE***************")
        #df=df.append(save_to_dataframe(llm_extracted_data), ignore_index=True)

    df.head()
    return df


# create streamlit app

st.set_page_config(page_title="Bill Extractor")
st.title("Extract Bill with AI 🤖")

# Upload Bills
pdf_files = st.file_uploader("Please upload your bills in PDF format",
                             type=["pdf"],
                             accept_multiple_files=True)
extract_button = st.button("Extracting bill ...")

if extract_button:
    with st.spinner("Extracting takes time..."):
        data_frame = create_docs(pdf_files)
        st.write(data_frame.head())
        data_frame["Total Amount Due"] = data_frame["Total Amount Due"].astype(float)
        st.write("Average Total Amount Due: ", data_frame['Total Amount Due'].mean())
        
        # convert to csv
        convert_to_csv = data_frame.to_csv(index=False).encode("utf-8")
        
        
        st.download_button(
            "CSV Download data",
            convert_to_csv,
            "CSV_Bills.csv",
            "text/csv",
            key="download-csv"
        )
    st.success("Successfully Done!!!")