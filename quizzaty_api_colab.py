import asyncio
import nest_asyncio
nest_asyncio.apply()

from typing import Union, Annotated
from fastapi import FastAPI, UploadFile, Form, File
from pyngrok import ngrok
import uvicorn
from pydantic import BaseModel
import time
import json
import re
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi.responses import JSONResponse
from math import ceil
import os
import shutil
import tempfile
from functools import partial

# Import llama_index libraries 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, PropertyGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.async_utils import asyncio_run

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
    doc = None
    query_engine = None
    
    def __init__(self):
        # Create temp directories that will be used throughout the lifecycle
        self.storage_dir = tempfile.mkdtemp()
        self.upload_dir = tempfile.mkdtemp()
        
    def __del__(self):
        # Clean up temporary directory when object is destroyed
        if self.storage_dir and os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)
        if self.upload_dir and os.path.exists(self.upload_dir):
            shutil.rmtree(self.upload_dir)

    # load the model if not loaded
    def load_model(self):
        model_name_questions = "deepseek-r1-distill-llama-70b"
        self.llm_questions = Groq(model=model_name_questions, api_key=self.llm_api,max_retries=2)
        self.embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return

    # load the uploaded document
    def load_doc(self, file, path):
        file_path = f'{self.upload_dir}/{file.filename}'
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Read documents from the temp directory
        documents = SimpleDirectoryReader(self.upload_dir).load_data()
        return documents
    
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
        self.index.storage_context.persist(persist_dir=self.storage_dir)
        return self.index

    # load the index
    def load_index(self, path):
        self.query_engine = self.index.as_query_engine(
            llm=self.llm_questions,
            embed_model=self.embedding_model,
            storage_context=self.index.storage_context
        )
        return
        
    # prediction
    async def prediction(self, difficulty_level):
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
        # Clean up both temporary directories
        if self.storage_dir and os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)
            self.storage_dir = tempfile.mkdtemp()
        if self.upload_dir and os.path.exists(self.upload_dir):
            shutil.rmtree(self.upload_dir)
            self.upload_dir = tempfile.mkdtemp()
        self.index = None
    
    def extract_json_from_response(self, response: str):
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
        
    def add_to_json(self, json_data, difficulty_str, chapter_number):
        # Map string to integer values
        difficulty_map = {
            "easy": 1,
            "medium": 2,
            "hard": 3
        }
    
        # Get the corresponding numeric value
        difficulty_value = difficulty_map.get(difficulty_str.lower(), 1)  # default to 1 if not found
    
        for item in json_data:
            item["difficulty"] = difficulty_value
            item["chapterNo"] = chapter_number
    
        return json_data

# create the app
app = FastAPI()

# create the graphRAG object
graphrag = graphRAG()

# get request
@app.get('/')
def index():
    return {'message': 'Quizaty API!'}

# post request that takes a file and returns questions
@app.post('/questions')
async def predict(file: Annotated[UploadFile, File()], chapter_number: int = Form(1)):
    try:
        path = "./"
        print(f"Received file: {file.filename}")
        print(f"Chapter number: {chapter_number}")
        
        try:
            print("Starting model loading...")
            graphrag.load_model()
            print("Model loading completed successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Model Loading Error", "step": "load_model", "details": str(e)}
            )

        try:
            print("Starting document loading...")
            document = graphrag.load_doc(file, path)
            print(f"Document loading completed. Number of documents: {len(document)}")
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Document Loading Error", "step": "load_doc", "details": str(e)}
            )

        try:
            print("Starting document indexing...")
            await graphrag.index_doc(document, path)
            print("Document indexing completed")
        except Exception as e:
            print(f"Error indexing document: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Document Indexing Error", "step": "index_doc", "details": str(e)}
            )

        try:
            print("Starting index loading...")
            graphrag.load_index(path)
            print("Index loading completed")
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Index Loading Error", "step": "load_index", "details": str(e)}
            )

        json_data_all = []
        for i in ["easy", "medium", "hard"]:
            try:
                print(f"Generating questions for {i} difficulty...")
                test = await graphrag.prediction(i)
                print(f"Generated {i} difficulty questions")
                response_answer = str(test)
                json_data = graphrag.extract_json_from_response(response_answer)
                print(f"Extracted {len(json_data)} questions from response")
                json_data = graphrag.add_to_json(json_data, i, chapter_number)
                json_data_all.extend(json_data)
                print(f"Completed {i} difficulty level")
                time.sleep(3)
            except Exception as e:
                print(f"Error generating {i} difficulty questions: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Question Generation Error for {i} difficulty",
                        "step": "prediction",
                        "details": str(e)
                    }
                )

        try:
            print("Cleaning up...")
            graphrag.clear_neo4j()
            print("Cleanup completed")
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")
            
        print(f"Successfully generated {len(json_data_all)} total questions")
        return JSONResponse(content=json_data_all)
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Error", "step": "general", "details": str(e)}
        )

def start_server():
    # Start ngrok tunnel
    port = 8000
    public_url = ngrok.connect(port).public_url
    print(f"Public URL: {public_url}")
    
    # Start the FastAPI app
    uvicorn.run(app, host="127.0.0.1", port=port)

if __name__ == "__main__":
    # Set your ngrok authentication token
    # You should set this in Google Colab with your actual token:
    # ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")
    
    # Start the server
    start_server() 