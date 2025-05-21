# Standard library imports
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
# from datetime import datetime
from typing import Union, Annotated
# from math import ceil
# from functools import partial

# Third-party imports
import nest_asyncio
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# from pyngrok import ngrok
# from tenacity import retry, stop_after_attempt, wait_exponential
# import uvicorn

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, PropertyGraphIndex, StorageContext, load_index_from_storage, SimpleDirectoryReader
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.async_utils import asyncio_run
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.graph_stores.falkordb import FalkorDBGraphStore
# from llama_index.graph_stores.falkordb import FalkorDBPropertyGraphStore
# from dotenv import load_dotenv

# Apply nest_asyncio
nest_asyncio.apply()

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
    llm_api_2 = "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"
    llm_api_3 = "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"
    llm_api_4 = "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"
    doc = None
    query_engine = None
    storage_dir = None
    upload_dir = None
    graph_store = None
    
    def __init__(self):
        # Create temp directories that will be used throughout the lifecycle
        self.storage_dir = tempfile.mkdtemp()
        self.upload_dir = tempfile.mkdtemp()
        
        try:
            # Connect to FalkorDB Docker container
            self.graph_store = FalkorDBGraphStore(
                "redis://0.0.0.0:6379",  # Docker container port
                decode_responses=True
            )
        except Exception as e:
            print(f"Warning: Could not connect to FalkorDB: {str(e)}")
            print("Falling back to in-memory graph store")
            # Fallback to in-memory graph store if FalkorDB is not available
            self.graph_store = SimpleGraphStore()

    def __del__(self):
        # Clean up temporary directories when object is destroyed
        for dir_path in [self.storage_dir, self.upload_dir]:
            if dir_path and os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Warning: Error cleaning up directory {dir_path}: {str(e)}")

    # load the model if not loaded
    def load_model(self):
        try:
            model_name_questions = "deepseek-r1-distill-llama-70b"
            # Initialize all LLM instances
            self.llm_graph_1 = Groq(model=model_name_questions, api_key=self.llm_api, max_retries=2)
            self.llm_graph_2 = Groq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=self.llm_api_2, max_retries=2)
            self.llm_graph_3 = Groq(model="gemma2-9b-it", api_key=self.llm_api_3, max_retries=2)
            self.llm_graph_4 = Groq(model="llama-guard-3-8b", api_key=self.llm_api_4, max_retries=2)
            
            # Set the main LLM instance (for backward compatibility)
            self.llm_questions = self.llm_graph_1
            
            self.embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            print("All LLM instances initialized successfully")
            return
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise

    # load the uploaded document
    def load_doc(self, file, path):
        file_path = os.path.join(self.upload_dir, file.filename)
        
        try:
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Read documents from the upload directory
            documents = SimpleDirectoryReader(self.upload_dir).load_data()
            return documents
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            raise

    async def index_doc(self, doc, path):
        print("Initializing shared SimpleGraphStore...")
        
        # Clear existing graph store
        try:
            self.graph_store.clear()
            print("Cleared existing graph store")
        except Exception as e:
            print(f"Warning: Could not clear graph store: {str(e)}")
        
        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        
        # Use the already initialized LLM instances with their model names
        llm_instances = [
            (self.llm_graph_1, "deepseek-r1-distill-llama-70b"),
            (self.llm_graph_2, "meta-llama/llama-4-scout-17b-16e-instruct"),
            (self.llm_graph_3, "gemma2-9b-it"),
            (self.llm_graph_4, "llama-guard-3-8b")
        ]
        
        # Split document into chunks
        doc_length = len(doc)
        total_chunks = 955  # Total number of chunks to process
        chunks_per_llm = total_chunks // 4  # Divide chunks among 4 LLMs
        overlap = 200  # Add some overlap between chunks
        chunks = []
        
        # Create chunks for each LLM
        for i in range(4):
            start_chunk = i * chunks_per_llm
            end_chunk = start_chunk + chunks_per_llm if i < 3 else total_chunks
            llm_chunks = []
            
            for j in range(start_chunk, end_chunk):
                start_idx = max(0, j * (doc_length // total_chunks) - overlap if j > 0 else 0)
                end_idx = min(doc_length, (j + 1) * (doc_length // total_chunks) + overlap if j < total_chunks - 1 else doc_length)
                llm_chunks.append(doc[start_idx:end_idx])
            
            chunks.append(llm_chunks)
        
        print(f"Split document into {len(chunks)} LLM groups")
        for i, (_, model_name) in enumerate(llm_instances):
            print(f"LLM {i+1} ({model_name}) will process {len(chunks[i])} chunks")
        
        # Process chunks in parallel using different LLM instances
        async def process_chunk(chunk, llm, model_name, chunk_id, llm_id, total_chunks):
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    print(f"\nLLM {llm_id} ({model_name}) - Processing chunk {chunk_id + 1}/{total_chunks}")
                    chunk_index = PropertyGraphIndex.from_documents(
                        chunk,
                        llm=llm,
                        embed_model=self.embedding_model,
                        storage_context=storage_context,
                        show_progress=False,  # Disable individual progress bars
                        use_async=True
                    )
                    print(f"LLM {llm_id} ({model_name}) - Completed chunk {chunk_id + 1}/{total_chunks}")
                    return chunk_index
                except Exception as e:
                    error_str = str(e)
                    if "RateLimitError" in error_str or "rate_limit_exceeded" in error_str:
                        retry_count += 1
                        if retry_count < max_retries:
                            print(f"\nLLM {llm_id} ({model_name}) - Rate limit hit. Waiting 60 seconds before retry {retry_count}/{max_retries}")
                            await asyncio.sleep(60)
                            continue
                    print(f"Error processing chunk {chunk_id} with LLM {llm_id} ({model_name}): {str(e)}")
                    raise
        
        # Create tasks for parallel processing with timeouts
        async def process_llm_chunks(llm_chunks, llm, model_name, llm_id):
            chunk_indices = []
            total_chunks = len(llm_chunks)
            
            for i, chunk in enumerate(llm_chunks):
                try:
                    # Process chunk
                    chunk_index = await process_chunk(chunk, llm, model_name, i, llm_id, total_chunks)
                    chunk_indices.append(chunk_index)
                    
                    # Add timeout after every 3 chunks
                    if (i + 1) % 3 == 0:
                        print(f"\nLLM {llm_id} ({model_name}) taking a 60-second break after chunk {i+1}/{total_chunks}")
                        await asyncio.sleep(60)
                    
                except Exception as e:
                    print(f"Error in LLM {llm_id} ({model_name}) processing chunk {i}: {str(e)}")
                    raise
            
            return chunk_indices
        
        # Create tasks for each LLM
        tasks = [
            process_llm_chunks(chunks[i], llm, model_name, i+1)
            for i, ((llm, model_name), _) in enumerate(zip(llm_instances, chunks))
        ]
        
        try:
            # Process all LLM groups in parallel
            all_chunk_indices = await asyncio.gather(*tasks)
            
            # Flatten the results
            chunk_indices = [idx for llm_indices in all_chunk_indices for idx in llm_indices]
            
            # Merge the indices (FalkorDB will handle this automatically)
            self.index = chunk_indices[0]  # Use the first index as the base
            
            print(f"\nProcessing complete with {len(chunk_indices)} total chunks")
            return self.index
        except Exception as e:
            print(f"Error during parallel processing: {str(e)}")
            # Fallback to single chunk processing if parallel processing fails
            print("Falling back to single chunk processing...")
            self.index = PropertyGraphIndex.from_documents(
                doc,
                llm=self.llm_questions,
                embed_model=self.embedding_model,
                storage_context=storage_context,
                show_progress=True,
                use_async=True
            )
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
        difficulty_value = difficulty_map.get(difficulty_str.lower(), 1)  # default to 0 if not found
    
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


# post request that takes a review (text type) and returns a sentiment score
@app.post('/questions')
async def predict(file: Annotated[UploadFile, File()]):
    try:
        path = "./"
        print(f"Received file: {file.filename}")
        print(f"Received path: {path}")
        
        # Initialize models
        try:
            graphrag.load_model()
            print("load_model : done")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "Model Loading Error", "step": "load_model", "details": str(e)}
            )

        # Load and process document
        try:
            document = graphrag.load_doc(file, path)
            print("load_doc : done")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "Document Loading Error", "step": "load_doc", "details": str(e)}
            )

        # Index document
        try:
            await graphrag.index_doc(document, path)
            print("index_doc : done")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "Document Indexing Error", "step": "index_doc", "details": str(e)}
            )

        # Load index
        try:
            graphrag.load_index(path)
            print("load_index : done")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "Index Loading Error", "step": "load_index", "details": str(e)}
            )

        # Generate questions for each difficulty level
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
                # Add a small delay between API calls to avoid rate limiting
                await asyncio.sleep(3)
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Question Generation Error for {i} difficulty",
                        "step": "prediction",
                        "details": str(e)
                    }
                )

        # Cleanup
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
