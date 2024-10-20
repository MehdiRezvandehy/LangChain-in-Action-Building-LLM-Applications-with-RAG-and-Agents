import openai
import os
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import streamlit as st    

os.environ["OPENAI_API_KEY"] = "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"

openai.api_key = os.getenv("OPENAI_API_KEY")  

# to find all environmental variables
load_dotenv(find_dotenv())

# OpenAI Chat API
model_llm = "gpt-3.5-turbo"

model_chat = ChatOpenAI(temperature=0.76, model=model_llm)
open_ai = OpenAI(temperature=0.78)


def lullaby_generate(location, name, language):

    template = """
    write an a fake stroy of 100 words for a person living in {location} 
    and make a living based on boxing. Make his/her name as {name} 
    
    fake STORY:
    """
    prompt = PromptTemplate(input_variables=["location", "name"],
                            template=template
                           )
    
    fake_story_chain = LLMChain(llm=open_ai, prompt=prompt, 
                                output_key="story",
                                verbose=True) # see what is going on in background
    #
    update_template = """
    # translate the {story} into {language}. Please ensure that the language is easily 
    understandable and is fun to read.
    
    Translation into {language}: 
    """
    
    translate_prompt = PromptTemplate(input_variables=["story", "language"],
                                     template=update_template)
    #
    translate_chain = LLMChain(llm=open_ai, 
                              prompt=translate_prompt, 
                              output_key="translated"
                             )                           
                             
    #
    chain_overall = SequentialChain(
        chains=[fake_story_chain, translate_chain],
        input_variables=["location", "name", "language"],
        output_variables=["story", "translated"], # This will return the story and translate it
        verbose=True
    )
    
    response = chain_overall({"location": location,
                            "name": name,
                            "language": language,
                             })                           
    
    return response
   
    
# Create a user interface here
def main():
    st.set_page_config(page_title="Generate a fake story",
                      layout="centered")
    st.title("Ask AI to write a fake story about a boxer and translate it to another language 📚")
    st.header("Now it is started ...")
    location_input = st.text_input(label="Location for the story")
    name_input = st.text_input(label="What is the name of character")
    language_input = st.text_input(label="Translate story to another language")
    
    submit_button = st.button("Submit")
    if location_input and name_input and language_input:
        if submit_button:
            with st.spinner("Generate a Fake story..."):
                response = lullaby_generate(location=location_input,
                                                    name=name_input,
                                                    language=language_input
                                                    )
                with st.expander("English version"):
                    st.write(response['story'])
                    
                with st.expander(f"{language_input} language"):
                    st.write(response['translated'])
                    
            st.success("Successfully done!")    
    

#Invoking main function
if __name__ == '__main__':
    main()