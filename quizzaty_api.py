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
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi.responses import JSONResponse

# response of the model
class Prediction(BaseModel):
    response_answer: str

# error response model
class ErrorResponse(BaseModel):
    error: str
    step: str
    details: str

# input of the model
class input_body(BaseModel):
    path: str


# graphRAG class
class graphRAG:
    # variables
    llm_graph_1 = None
    llm_graph_2 = None
    llm_graph_3 = None
    llm_graph_4 = None
    llm_graph= None
    llm_questions = None
    embedding_model = None
    index = None
    llm_api = "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"
    # deepseek_r1_distill_llama_70b = "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"
    # questions_beshoy_1 = "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"
    # questions_beshoy_2 = "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"
    # questions_kerolos_1 = "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"
    # questions_kerolos_2 = "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"
    doc = None
    query_engine = None
    temp_dir = None
    
    def __init__(self):
        # Create a temporary directory for storage
        self.temp_dir = tempfile.mkdtemp()
        
    def __del__(self):
        # Clean up temporary directory when object is destroyed
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    # load the model if not loaded
    def load_model(self):
        # model_name_questions = "deepseek-r1-distill-llama-70b"
        model_name_questions = "llama3-8b-8192"
        self.llm_questions = Groq(model=model_name_questions, api_key=self.llm_api,max_retries=0)
        self.embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # model_name_graph = "llama3-8b-8192"
        # self.llm_graph_1 = Groq(model=model_name_graph, api_key=self.questions_beshoy_1)
        # self.llm_graph_2 = Groq(model=model_name_graph, api_key=self.questions_kerolos_1)
        # self.llm_graph_3 = Groq(model=model_name_graph, api_key=self.questions_beshoy_2)
        # self.llm_graph_4 = Groq(model=model_name_graph, api_key=self.questions_kerolos_2)
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

    # async def index_doc(self, doc, path):
    #     loop = asyncio.get_event_loop()

    #     # Split the document into 5 chunks
    #     chunk_size = ceil(len(doc) / 8)
    #     doc_chunks = [doc[i:i + chunk_size] for i in range(0, len(doc), chunk_size)]

    #     def create_index_shared_store():
    #         print("Initializing shared SimpleGraphStore...")
    #         # Create a shared in-memory graph store
    #         shared_graph_store = SimpleGraphStore()
    #         shared_storage_context = StorageContext.from_defaults(graph_store=shared_graph_store)

    #         for i, chunk in enumerate(doc_chunks):
    #             print(f"Processing chunk {i}/8 — length: {len(chunk)}")
    #             if(i == 0 or i==4):
    #                 # Use the first LLM for even chunks
    #                 self.llm_graph = self.llm_graph_1
    #             elif(i == 1 or i==5):
    #                 # Use the second LLM for odd chunks
    #                 self.llm_graph = self.llm_graph_2
    #             elif(i == 2 or i==6):
    #                 # Use the third LLM for even chunks
    #                 self.llm_graph = self.llm_graph_3
    #             else:
    #                 # Use the fourth LLM for odd chunks
    #                 self.llm_graph = self.llm_graph_4
    #             # Each chunk writes into the same graph store
    #             PropertyGraphIndex.from_documents(
    #                 chunk,
    #                 llm=self.llm_graph,
    #                 embed_model=self.embedding_model,
    #                 storage_context=shared_storage_context,
    #                 show_progress=True,
    #             )
    #             time.sleep(60)  # Optional: Sleep to avoid overwhelming the api (rate limiting)
    #         # After all chunks processed, create one final index
    #         final_index = PropertyGraphIndex(
    #             storage_context=shared_storage_context,
    #             llm=self.llm_graph,
    #             embed_model=self.embedding_model,
    #         )

    #         return final_index

    #     self.index = await loop.run_in_executor(None, create_index_shared_store)

    #     self.index.storage_context.persist(persist_dir="./storage")
    #     return self.index
    
    async def index_doc(self, doc, path):
        print("Initializing shared SimpleGraphStore...")
        # Create a shared in-memory graph store
        shared_graph_store = SimpleGraphStore()
        shared_storage_context = StorageContext.from_defaults(graph_store=shared_graph_store)
        self.index = PropertyGraphIndex.from_documents(
                doc,
                llm=self.llm_questions,
                embed_model=self.embedding_model,
                storage_context=shared_storage_context,
                show_progress=True,
                use_async=True
            )
        self.index.storage_context.persist(persist_dir="./storage")
        return self.index

    # load the index
    def load_index(self,path):
        self.query_engine = self.index.as_query_engine(
            llm=self.llm_questions,
        )
        return
        
    # prediction
    async def prediction(self,difficulty_level):  # dependency
        # difficulty_level = "easy"
        response = self.query_engine.query(f"""You are an AI designed to generate multiple-choice questions (MCQs) based on a provided chapter of a book. Your task is to create a set of MCQs that focus on the main subject matter of the chapter. Ensure that each question is clear, concise, and relevant to the core themes of the chapter and be closed book style. Use the following structure for the MCQs:
            
            1. **Question Statement**: A clear and precise question related to the chapter content.
            2. **Answer Choices**: Four options labeled A, B, C, and D, where only one option is correct. The incorrect options should be plausible to challenge the reader's knowledge.
            3. **Correct Answer**: give me the correct answer of the question
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
            
            Please ensure that the questions reflect a deep understanding of the chapter's main ideas and concepts while varying the complexity to accommodate different levels of knowledge. Provide 40 questions for {difficulty_level} level. 
            
            Begin by analyzing the chapter content thoroughly to extract key concepts, terms, and themes that can be transformed into question formats. 
            
            End the generated MCQs with a summary statement of the chapter's main subject to reinforce learning.
            and make the output form in json form like thie example : 
            {{
            "question":"Which of the following is not one of the characteristics of a data communication system?",
            "answerA":"Delivery",
            "answerB":"Accuracy",
            "answerC":"Jitter",
            "answerD":"All of the choices are correct",
            "correctAnswer":"answerD"
            }}""")
        return response

    # clear the index
    def clear_neo4j(self):
        # Clear the index
        if self.index:
            shutil.rmtree("./storage")
            self.index = None
    
    def extract_json_from_response(self,response: str):
        # Match individual JSON objects inside brackets
        object_matches = re.findall(r'{[^{}]*?(?:"[^"]*":\s*"[^"]*?",?)*[^{}]*?}', response, re.DOTALL)
    
        valid_objects = []
        for obj_str in object_matches:
            try:
                obj = json.loads(obj_str)
                valid_objects.append(obj)
            except json.JSONDecodeError:
                continue  # Skip invalid or incomplete objects
    
        return valid_objects
    def add_to_json(self ,json_data, difficulty_str, chapter_number):
        # Map string to integer values
        difficulty_map = {
            "easy": 1,
            "medium": 2,
            "hard": 3
        }
    
        # Get the corresponding numeric value
        difficulty_value = difficulty_map.get(difficulty_str.lower(), 1)  # default to 0 if not found
    
        for item in json_data:
            item["difficulty"] = difficulty_value
            item["chapterNo"] = chapter_number
    
        return json_data

# import the libraries
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import PropertyGraphIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.async_utils import asyncio_run

import os
from datetime import datetime
import tempfile
import shutil
import json
import re
from math import ceil

import asyncio
from functools import partial

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
async def predict(file: Annotated[UploadFile, File()]):
    try:
        path = "./"
        print(f"Received file: {file.filename}")
        print(f"Received path: {path}")
        
        try:
            graphrag.load_model()
            print("load_model : done")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "Model Loading Error", "step": "load_model", "details": str(e)}
            )

        try:
            document = graphrag.load_doc(file, path)
            print("load_doc : done")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "Document Loading Error", "step": "load_doc", "details": str(e)}
            )

        try:
            # graphrag.index_doc(document, path)
            asyncio_run(graphrag.index_doc(document, path))
            print("index_doc : done")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "Document Indexing Error", "step": "index_doc", "details": str(e)}
            )

        try:
            graphrag.load_index(path)
            print("load_index : done")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "Index Loading Error", "step": "load_index", "details": str(e)}
            )

        json_data_all = []
        for i in ["easy", "medium", "hard"]:
            try:
                test = await graphrag.prediction(i)
                print(type(test))
                response_answer = str(test)
                json_data = graphrag.extract_json_from_response(response_answer)
                print("extract_json_from_response : done")
                json_data = graphrag.add_to_json(json_data, i, 1)
                print("add to json : done")
                json_data_all.extend(json_data)
                print(f"difficulty {i}: done")
                time.sleep(3)
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Question Generation Error for {i} difficulty",
                        "step": "prediction",
                        "details": str(e)
                    }
                )

        try:
            graphrag.clear_neo4j()
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")
            # Don't return error for cleanup issues, just log it
            
        return JSONResponse(content=json_data_all)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Error", "step": "general", "details": str(e)}
        )


