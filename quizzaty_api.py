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
import google.generativeai as genai

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, PropertyGraphIndex, StorageContext, load_index_from_storage, SimpleDirectoryReader, Document
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.async_utils import asyncio_run
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.llms.together import TogetherLLM
from llama_index.graph_stores.falkordb import FalkorDBGraphStore
from llama_index.core.node_parser import TokenTextSplitter

from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core import Settings

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
    llm_1 = None
    llm_2 = None
    llm_3 = None
    llm_graph_4 = None
    llm_graph= None
    llm_questions = None
    embedding_model = None
    index = None
    # llm_api = "tgp_v1_mBgpHIOQ76SvIKx6I5OhOcREZFWkEiDJAU4FdZ4qAzE"  # Replace with your Together AI API key
    llm_api = "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"
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
        # Configure Gemini with API key from environment variable or fallback to hardcoded key

        self.llm_gemini = Gemini(
            model="gemini-2.0-flash-lite",
            api_key="AIzaSyAwuVnbkTAMhR5-DxwYzwBN9-vilX_bnXY",
            max_retries=2
        )
        self.llm_groq = Groq(
            model="llama-3.1-8b-instant",
            api_key="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",
            max_retries=2
        )
        self.llm_together = TogetherLLM(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            api_key="tgp_v1_l2VKQAmfsfOF3R6jht38IvlWFwZ7bzEkqm2GVfjUL14",
            max_retries=2
        )
        # self.llm_3 = Gemini(
        #     model=model_name,
        #     api_key="AIzaSyBK1p3akSoS5ioEuMfuYD4Bq7K7pXqKnjw",
        #     max_retries=2
        # )
        # self.llm_4 = Gemini(
        #     model=model_name,
        #     api_key="AIzaSyB09dUhjrfIJ7QPrsnV585S5b9z1KkuwS4",
        #     max_retries=2
        # )
        # self.llm_5 = Gemini(
        #     model=model_name,
        #     api_key="AIzaSyAhoR1sPdMlRUwoKmD5jp4UUx131KKamKU",
        #     max_retries=2
        # )
        # self.llm_6 = Gemini(
        #     model=model_name,
        #     api_key="AIzaSyDmEd3JRN6qv_0jfm4cHg5wHVrpAFiJYqE",
        #     max_retries=2
        # )
        # self.llm_7 = Gemini(
        #     model=model_name,
        #     api_key="AIzaSyBtarGCXK7q_1PnZ3DBmzfyn-8EpCgl_VM",
        #     max_retries=2
        # )
        # self.llm_8 = Gemini(
        #     model=model_name,
        #     api_key="AIzaSyCoK5ND4racg4JpTcBgNUT_r9J2-clzjNA",
        #     max_retries=2
        # )
        # self.llm_9 = Gemini(
        #     model=model_name,
        #     api_key="AIzaSyBqGHbJr8e5MDG6cKmwQJw_xMydXxBQMXU",
        #     max_retries=2
        # )


        self.embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        Settings.chunk_size = 500
        Settings.chunk_overlap = 200

        return

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
                # Set global settings
        # Settings.num_workers = 4
        # Settings.llm = self.llm_questions
        # self.index = PropertyGraphIndex.from_documents(
        #     doc,
        #     llm=self.llm_questions,
        #     embed_model=self.embedding_model,
        #     storage_context=storage_context,  # Use the created storage context
        #     show_progress=True,
        #     num_workers=2,
        #     chunk_size=1024,  # Process 1024 tokens per chunk
        #     chunk_overlap=100,  # 300 token overlap between chunks
        #     chunk_sleep_time=90.0  # Sleep 1 second between chunks
        # )
        print("Initializing shared SimpleGraphStore...")
        # Create storage context with the graph store
        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        
        # Initialize text splitter with specified chunk size and overlap
        text_splitter = TokenTextSplitter(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap
        )
        
        # Process each document and split into chunks
        chunked_docs = []
        for document in doc:
            try:
                # Clean the text before splitting
                cleaned_text = re.sub(r'[^\x00-\x7F]+', '', document.text)  # Remove non-ASCII characters
                chunks = text_splitter.split_text(cleaned_text)
                # Convert chunks back to Document objects
                doc_chunks = [Document(text=chunk) for chunk in chunks]
                chunked_docs.extend(doc_chunks)
            except Exception as e:
                print(f"Error processing document: {str(e)}")
                continue
            
        # Replace original documents with chunked version
        doc = chunked_docs
        print(f"Total chunks to process: {len(doc)}")
        
        # Start timer
        start_time = time.time()
        
        # Create a list of LLMs to use with their names
        llms = [
            (self.llm_groq, "groq"),
            # (self.llm_together, "together"),
            (self.llm_gemini, "gemini"),
        ]
        
        # Dictionary to track processing times for each LLM
        llm_processing_times = {name: [] for _, name in llms}
        llm_chunk_counts = {name: 0 for _, name in llms}
        llm_request_counts = {name: 0 for _, name in llms}
        
        async def process_chunk(chunk, llm_tuple, chunk_index):
            llm, llm_name = llm_tuple
            chunk_start = time.time()
            try:
                print(f"\n{llm_name} starting to process chunk {chunk_index}")
                
                # Clean the chunk text before processing
                cleaned_chunk = Document(text=re.sub(r'[^\x00-\x7F]+', '', chunk.text))
                
                # Create PropertyGraphIndex with async support
                result = PropertyGraphIndex.from_documents(
                    [cleaned_chunk],
                    llm=llm,
                    embed_model=self.embedding_model,
                    storage_context=storage_context,
                    use_async=True,
                    num_workers=1,  # Reduced number of workers
                    # show_progress=True
                )
                
                chunk_end = time.time()
                processing_time = chunk_end - chunk_start
                llm_processing_times[llm_name].append(processing_time)
                llm_chunk_counts[llm_name] += 1
                llm_request_counts[llm_name] += 1
                print(f"{llm_name} completed chunk {chunk_index} in {processing_time:.2f} seconds (Total requests: {llm_request_counts[llm_name]})")
                return result
            except RecursionError as e:
                print(f"Recursion error in {llm_name} processing chunk {chunk_index}: {str(e)}")
                return None
            except Exception as e:
                chunk_end = time.time()
                processing_time = chunk_end - chunk_start
                print(f"Error in {llm_name} processing chunk {chunk_index}: {str(e)}")
                print(f"Failed chunk took {processing_time:.2f} seconds")
                return None

        async def process_batch(batch_chunks, llm_tuple, batch_number):
            llm, llm_name = llm_tuple
            batch_start = time.time()
            print(f"\n{llm_name} starting batch {batch_number} with {len(batch_chunks)} chunks")
            
            # Process chunks concurrently using asyncio.gather
            tasks = []
            for j, chunk in enumerate(batch_chunks):
                chunk_index = (batch_number * len(batch_chunks)) + j
                tasks.append(process_chunk(chunk, llm_tuple, chunk_index))
            
            # Wait for all chunks in the batch to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update index with the last successful result
            for result in results:
                if isinstance(result, Exception):
                    print(f"Error in batch processing: {str(result)}")
                    continue
                if result is not None:
                    self.index = result
            
            batch_end = time.time()
            batch_duration = batch_end - batch_start
            print(f"\n{llm_name} completed batch {batch_number} in {batch_duration:.2f} seconds")
            return batch_duration

        # Calculate chunks per batch
        chunks_per_batch = 25  # Reduced batch size
        total_batches = (len(doc) + chunks_per_batch - 1) // chunks_per_batch
        
        # Process batches in parallel, with each LLM handling its own batch
        round_counter = 0
        for batch_start in range(0, total_batches, len(llms)):
            round_counter += 1
            round_requests = {llm_name: 0 for _, llm_name in llms}
            batch_tasks = []
            for i, llm_tuple in enumerate(llms):
                batch_number = batch_start + i
                if batch_number >= total_batches:
                    continue
                    
                start_idx = batch_number * chunks_per_batch
                end_idx = min(start_idx + chunks_per_batch, len(doc))
                batch_chunks = doc[start_idx:end_idx]
                
                batch_tasks.append(process_batch(batch_chunks, llm_tuple, batch_number))
            
            # Wait for all batches in this round to complete
            batch_durations = await asyncio.gather(*batch_tasks)
            
            # Print round summary
            print(f"\nRound {round_counter} Summary:")
            for i, duration in enumerate(batch_durations):
                llm_name = llms[i][1]
                # Calculate requests for this round
                round_requests[llm_name] = llm_request_counts[llm_name] - sum(round_requests.values())
                print(f"{llm_name} processed batch {batch_start + i} in {duration:.2f} seconds (Round requests: {round_requests[llm_name]}, Total requests: {llm_request_counts[llm_name]})")
            
            # Sleep between rounds if there are more batches to process
            if batch_start + len(llms) < total_batches:
                print("\nSleeping for 60 seconds before next round...")
                await asyncio.sleep(60)  # Reduced sleep time
        
        # End timer and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        
        # Print final statistics
        print("\nFinal Processing Statistics:")
        print(f"Total processing time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print("\nLLM Performance Summary:")
        for llm_name in llm_processing_times:
            if llm_chunk_counts[llm_name] > 0:
                total_time = sum(llm_processing_times[llm_name])
                avg_time = total_time / len(llm_processing_times[llm_name])
                print(f"{llm_name}:")
                print(f"  - Total chunks processed: {llm_chunk_counts[llm_name]}")
                print(f"  - Total requests made: {llm_request_counts[llm_name]}")
                print(f"  - Total processing time: {total_time:.2f} seconds")
                print(f"  - Average time per chunk: {avg_time:.2f} seconds")
        
        return self.index

    # load the index
    def load_index(self, path):
        try:
            # Create query engine for PropertyGraphIndex
            self.query_engine = self.index.as_query_engine(
                llm=self.llm_gemini,
                storage_context=self.index.storage_context,
                include_text=True,  # Include the text in the response
                response_mode="tree_summarize"  # Use tree summarization for better responses
            )
            return
        except Exception as e:
            print(f"Error in load_index: {str(e)}")
            raise
        
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
