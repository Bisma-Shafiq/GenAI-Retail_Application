from few_shot import few_shots
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
import pymysql
from langchain_experimental.sql import SQLDatabaseChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts import PromptTemplate, SemanticSimilarityExampleSelector, FewShotPromptTemplate
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define the BaseCache model first
class BaseCache(BaseModel):
    query_history: Dict[str, Any] = Field(default_factory=dict)
    
    def get_cache(self, query: str) -> Optional[Any]:
        return self.query_history.get(query)
    
    def update_cache(self, query: str, result: Any) -> None:
        self.query_history[query] = result

# Initialize cache globally
CACHE = BaseCache()

def get_chain_db():
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Initialize the ChatGoogleGenerativeAI model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
    
    # Database connection settings
    user = "root"
    password = "bisma123"
    host = "localhost"
    name = "ecommerce"
    
    # Connect to the MySQL database
    database = SQLDatabase.from_uri(
        f"mysql+pymysql://{user}:{password}@{host}/{name}",
        sample_rows_in_table_info=3
    )
    
    # Set up embeddings and vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    vectorize = [" ".join(str(v) for v in example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(vectorize, embedding=embeddings, metadatas=few_shots)
    
    # Example selector for similarity-based few-shot learning
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=3
    )

    # Define the prompt template
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}"
    )

    # Define the few-shot prompt
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"]
    )

    # Create the database chain
    db_chain = SQLDatabaseChain(
        llm=llm,
        database=database,
        prompt=few_shot_prompt,
        verbose=True,
        use_query_checker=True
    )
    
    return db_chain

def query_database(question: str) -> str:
    """
    Query the database with caching support
    """
    # Check cache first
    cached_result = CACHE.get_cache(question)
    if cached_result is not None:
        return cached_result
    
    # If not in cache, create chain and query
    chain = get_chain_db()
    try:
        result = chain.run(question)
        # Update cache
        CACHE.update_cache(question, result)
        return result
    except Exception as e:
        return f"Error processing query: {str(e)}"