import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

# load groq and google api key from env

groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.header("Gemma-Model Document Q&A")


llm = ChatGroq(groq_api_key=groq_api_key,model_name ="Gemma-7b-it")
