# Standard library imports
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from typing import Union, Annotated

# Third-party imports
import nest_asyncio
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, PropertyGraphIndex, StorageContext, load_index_from_storage, SimpleDirectoryReader, Document
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.async_utils import asyncio_run
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.graph_stores.falkordb import FalkorDBGraphStore
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Settings

# Apply nest_asyncio
nest_asyncio.apply()

class RateLimiter:
    def __init__(self, max_tokens_per_minute, max_requests_per_minute):
        self.max_tokens = max_tokens_per_minute
        self.max_requests = max_requests_per_minute
        self.tokens = 0
        self.requests = 0
        self.last_reset = time.time()

    async def acquire(self, tokens_needed):
        now = time.time()
        if now - self.last_reset >= 60:
            self.tokens = 0
            self.requests = 0
            self.last_reset = now
        while True:
            if (self.requests < self.max_requests and 
                self.tokens + tokens_needed <= self.max_tokens):
                self.requests += 1
                self.tokens += tokens_needed
                return
            else:
                wait_time = max(60 - (now - self.last_reset), 0)
                await asyncio.sleep(wait_time)
                self.tokens = 0
                self.requests = 0
                self.last_reset = time.time()

class Prediction(BaseModel):
    response_answer: str

class ErrorResponse(BaseModel):
    error: str
    step: str
    details: str

class input_body(BaseModel):
    path: str

class graphRAG:
    def __init__(self):
        self.storage_dir = tempfile.mkdtemp()
        self.upload_dir = tempfile.mkdtemp()
        try:
            self.graph_store = FalkorDBGraphStore(
                "redis://0.0.0.0:6379",
                decode_responses=True
            )
        except Exception as e:
            print(f"Warning: Falling back to in-memory graph store: {str(e)}")
            self.graph_store = SimpleGraphStore()

        # Initialize rate limiters for each LLM
        self.rate_limiters = {
            "llm_1": RateLimiter(max_tokens=60000, max_requests=30),
            "llm_2": RateLimiter(max_tokens=60000, max_requests=30),
            "llm_3": RateLimiter(max_tokens=60000, max_requests=30),
            "llm_questions": RateLimiter(max_tokens=60000, max_requests=30)
        }

    def __del__(self):
        for dir_path in [self.storage_dir, self.upload_dir]:
            if dir_path and os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Warning: Error cleaning up directory: {str(e)}")

    def load_model(self):
        model_name = "gemma-3-27b-it"
        self.llm_questions = Gemini(
            model=model_name,
            api_key="AIzaSyAwuVnbkTAMhR5-DxwYzwBN9-vilX_bnXY",
            max_retries=2
        )
        self.llm_1 = Gemini(
            model=model_name,
            api_key="AIzaSyBQfIuQshM7o4aM2t3kxC3bie67eCGG3Kk",
            max_retries=2
        )
        self.llm_2 = Gemini(
            model=model_name,
            api_key="AIzaSyDgFA3k1ayTmqzuEzuFKCpGlXKko9otX6o",
            max_retries=2
        )
        self.llm_3 = Gemini(
            model=model_name,
            api_key="AIzaSyBK1p3akSoS5ioEuMfuYD4Bq7K7pXqKnjw",
            max_retries=2
        )
        self.embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.chunk_size = 500
        Settings.chunk_overlap = 200

    def load_doc(self, file, path):
        file_path = os.path.join(self.upload_dir, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            return SimpleDirectoryReader(self.upload_dir).load_data()
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            raise

    async def index_doc(self, doc, path):
        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        text_splitter = TokenTextSplitter(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap
        )
        
        chunked_docs = []
        for document in doc:
            chunks = text_splitter.split_text(document.text)
            chunked_docs.extend([Document(text=chunk) for chunk in chunks])
        
        doc = chunked_docs
        print(f"Total chunks to process: {len(doc)}")

        chunk_queue = asyncio.Queue()
        for chunk in doc:
            await chunk_queue.put(chunk)

        llm_mapping = {
            "llm_1": self.llm_1,
            "llm_2": self.llm_2,
            "llm_3": self.llm_3,
            "llm_questions": self.llm_questions
        }

        async def worker(llm_name, worker_id):
            llm = llm_mapping[llm_name]
            limiter = self.rate_limiters[llm_name]
            while not chunk_queue.empty():
                chunk = await chunk_queue.get()
                try:
                    await limiter.acquire(tokens_needed=500)
                    result = await asyncio.to_thread(
                        PropertyGraphIndex.from_documents,
                        [chunk],
                        llm=llm,
                        embed_model=self.embedding_model,
                        storage_context=storage_context
                    )
                    if result:
                        self.index = result
                    print(f"Worker {worker_id} ({llm_name}) processed chunk")
                except Exception as e:
                    print(f"Worker {worker_id} failed: {e}")
                    await chunk_queue.put(chunk)
                finally:
                    chunk_queue.task_done()

        tasks = []
        for idx, llm_name in enumerate(["llm_1", "llm_2", "llm_3", "llm_questions"]):
            task = asyncio.create_task(worker(llm_name, idx))
            tasks.append(task)

        await chunk_queue.join()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        return self.index

    # Remaining methods (load_index, prediction, etc.) remain the same
    # ... [Keep the existing load_index, prediction, extract_json_from_response, 
    #      add_to_json, and clear_neo4j methods unchanged] ...

app = FastAPI()
graphrag = graphRAG()

@app.get('/')
def index():
    return {'message': 'Quizaty API!'}

@app.post('/questions')
async def predict(file: Annotated[UploadFile, File()]):
    try:
        path = "./"
        graphrag.load_model()
        document = graphrag.load_doc(file, path)
        await graphrag.index_doc(document, path)
        graphrag.load_index(path)
        
        json_data_all = []
        for i in ["easy", "medium", "hard"]:
            test = await graphrag.prediction(i)
            response_answer = str(test)
            json_data = graphrag.extract_json_from_response(response_answer)
            json_data = graphrag.add_to_json(json_data, i, 1)
            json_data_all.extend(json_data)
            await asyncio.sleep(3)

        graphrag.clear_neo4j()
        return JSONResponse(content=json_data_all)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Error", "step": "general", "details": str(e)}
        )