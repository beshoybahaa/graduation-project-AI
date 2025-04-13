import asyncio
import nest_asyncio
nest_asyncio.apply()

from typing import Union
from fastapi import FastAPI, UploadFile, Form, File
from pyngrok import ngrok
import uvicorn
from pydantic import BaseModel
from typing import Annotated
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# response of the model
class Prediction(BaseModel):
    response_answer: str

# input of the model
class input_body(BaseModel):
    path:str


# graphRAG class
class graphRAG:
    # variables
    llm_graph = None
    llm_questions = None
    embedding_model = None
    index = None
    deepseek_r1_distill_llama_70b = "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"
    gemma2_9b_it = "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"
    doc = None
    query_engine = None
    store = None
    
    # Create a Neo4jPropertyGraphStore instance
    def __init__(self):
        graph_store = Neo4jPropertyGraphStore(
            username="neo4j",
            password="g6DTWAKPPHTvJWNBEZ4vgTDSTt99ZUkE-hlyWv7-1Bg",
            url="neo4j+s://8fef1a11.databases.neo4j.io",
            #database="Instance01"
        )
        self.store = graph_store

    # load the model if not loaded
    def load_model(self):
        model_name_questions = "deepseek-r1-distill-llama-70b"
        self.llm_questions = Groq(model=model_name_questions, api_key=self.deepseek_r1_distill_llama_70b,max_retries=0)
        self.embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        model_name_graph = "gemma2-9b-it"
        self.llm_graph = Groq(model=model_name_graph, api_key=self.gemma2_9b_it)
        return

    # load the uploaded document
    def load_doc(self,file,path):
        file_path = f'{path}/{file.filename}.pdf'
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Read documents from the temp directory
        documents = SimpleDirectoryReader(path).load_data()
        return documents

    # index the document
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def index_doc(self,doc,path):
        try:
            self.index = PropertyGraphIndex.from_documents(
                doc,
                llm=self.llm_graph,
                embed_model=self.embedding_model,
                property_graph_store=self.store,
            )
            return self.index
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                print(f"Rate limit exceeded. Retrying...")
                raise
            else:
                print(f"Error during indexing: {str(e)}")
                raise

    # load the index
    def load_index(self,path):
        #self.index = load_index_from_storage(
        #    StorageContext.from_defaults(persist_dir=path),
        #   embed_model=self.embedding_model,
        #   llm=self.llm
        #)
        self.query_engine = self.index.as_query_engine(
            llm=self.llm_questions,
            include_text=True,
        )
        return
        
    # prediction
    def prediction(self):  # dependency
        response = self.query_engine.query("""You are an AI designed to generate multiple-choice questions (MCQs) based on a provided chapter of a book. Your task is to create a set of MCQs that focus on the main subject matter of the chapter, incorporating a range of difficulty levels. Ensure that each question is clear, concise, and relevant to the core themes of the chapter and be closed book style. Use the following structure for the MCQs:
            
            1. **Question Statement**: A clear and precise question related to the chapter content.
            2. **Answer Choices**: Four options labeled A, B, C, and D, where only one option is correct. The incorrect options should be plausible to challenge the reader's knowledge.
            3. **Difficulty Level**: Indicate the difficulty level of each question (Easy, Medium, Hard) to help categorize the questions.
            examples for questions : 
            1		Which of the following is not one of the components of a data communication system?
            	A)	Message
            	B)	Sender
            	C)	Medium
            	D)	All of the choices are correct
            
            2		Which of the following is not one of the characteristics of a data communication system?
            	A)	Delivery
            	B)	Accuracy
            	C)	Jitter
            	D)	All of the choices are correct
            
            Please ensure that the questions reflect a deep understanding of the chapter's main ideas and concepts while varying the complexity to accommodate different levels of knowledge. Provide 40 questions for hard level. 
            
            Begin by analyzing the chapter content thoroughly to extract key concepts, terms, and themes that can be transformed into question formats. 
            
            End the generated MCQs with a summary statement of the chapter's main subject to reinforce learning.""")
        return response

    # clear the neo4j
    def clear_neo4j(self):
        # Use the internal driver to execute raw Cypher
        with self.store._driver.session(database=self.store._database) as session:
            session.run("MATCH (n) DETACH DELETE n")

# import the libraries
from llama_index.core import PropertyGraphIndex
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from datetime import datetime
from llama_index.core import StorageContext, load_index_from_storage
import tempfile
import shutil
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# create the app
app = FastAPI()

# create the graphRAG object
graphrag = graphRAG()

# get request
@app.get('/')
def index():
    return {'message': 'Quizaty API!'}


# post request that takes a review (text type) and returns a sentiment score
@app.post('/questions')
async def predict(file: Annotated[UploadFile, File()]) -> Prediction:
    path = "./"
    print(f"Received file: {file.filename}")
    print(f"Received path: {path}")
    graphrag.load_model()
    print("load_model : done")
    document = graphrag.load_doc(file,path)
    print("load_doc : done")
    graphrag.index_doc(document,path)
    print("index_doc : done")
    graphrag.load_index(path)
    print("load_index : done")
    test= graphrag.prediction()
    print(type(test))
    response_answer = str(test)
    graphrag.clear_neo4j()
    return Prediction(response_answer=response_answer)


# load the model asynchronously on startup
# @app.on_event("startup")
# async def startup():
#     sentiment_model.load_model()


