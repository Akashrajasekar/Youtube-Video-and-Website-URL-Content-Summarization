import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
from dotenv import load_dotenv
# validators is an external library used to validate an url
# youtube_transcript_api gives transcript of youtube url entered

load_dotenv()
# Streamlit App
st.set_page_config(page_title="LangChain: YouTube Video and Website URL Content Summarization", page_icon="📄")
st.title("📄 LangChain: YouTube Video and Website URL Content Summarization")
st.subheader('URL for Summarization')

# Groq API Key and URL field
api_key=os.getenv("GROQ_API_KEY")
generic_url=st.text_input("URL",label_visibility="collapsed")

# Gemma Model Using Groq API
llm =ChatGroq(model="Gemma2-9b-It", groq_api_key=api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YouTube or Website URL"):
    ## Validate all the inputs
    if not api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YouTube video URL or Website URL")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or YouTube video data
                if "YouTube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)


                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")