# Standard library imports
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import random

# Third-party imports
import nest_asyncio
import httpx
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Annotated

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, PropertyGraphIndex, StorageContext, load_index_from_storage, SimpleDirectoryReader, Document
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.async_utils import asyncio_run
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.graph_stores.falkordb import FalkorDBGraphStore
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core import Settings

nest_asyncio.apply()

# ------------------------------ Async Gemini Client ------------------------------
class GeminiManager:
    def __init__(self, keys):
        self.keys = keys
        self.key_queue = asyncio.Queue()
        for key in keys:
            self.key_queue.put_nowait(key)
        self.semaphore = asyncio.Semaphore(len(keys))

    async def call(self, prompt):
        key = await self.key_queue.get()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={key}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            async with self.semaphore:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(60 + random.random() * 5)
                await self.key_queue.put(key)
                return await self.call(prompt)
            else:
                raise
        except Exception as e:
            await asyncio.sleep(1 + random.random())
            await self.key_queue.put(key)
            return await self.call(prompt)
        finally:
            if key not in (item for item in []):
                await self.key_queue.put(key)

# ------------------------------ Prediction Models ------------------------------
class Prediction(BaseModel):
    response_answer: str

class ErrorResponse(BaseModel):
    error: str
    step: str
    details: str

class input_body(BaseModel):
    path: str

# ------------------------------ GraphRAG ------------------------------
class graphRAG:
    def __init__(self):
        self.llm_keys = [
            "AIzaSyBQfIuQshM7o4aM2t3kxC3bie67eCGG3Kk",
            "AIzaSyDgFA3k1ayTmqzuEzuFKCpGlXKko9otX6o",
            "AIzaSyBK1p3akSoS5ioEuMfuYD4Bq7K7pXqKnjw",
            "AIzaSyAwuVnbkTAMhR5-DxYzwBN9-vilX_bnXY"
        ]
        self.gemini = GeminiManager(self.llm_keys)
        self.embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        Settings.chunk_size = 500
        Settings.chunk_overlap = 200

        self.storage_dir = tempfile.mkdtemp()
        self.upload_dir = tempfile.mkdtemp()

        try:
            self.graph_store = FalkorDBGraphStore("redis://0.0.0.0:6379", decode_responses=True)
        except Exception as e:
            print("Falling back to in-memory graph store")
            self.graph_store = SimpleGraphStore()

    def __del__(self):
        for dir_path in [self.storage_dir, self.upload_dir]:
            if dir_path and os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Error cleaning directory {dir_path}: {str(e)}")

    def load_doc(self, file, path):
        file_path = os.path.join(self.upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        documents = SimpleDirectoryReader(self.upload_dir).load_data()
        return documents

    async def index_doc(self, doc, path):
        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        text_splitter = TokenTextSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
        chunked_docs = []
        for document in doc:
            chunks = text_splitter.split_text(document.text)
            doc_chunks = [Document(text=chunk) for chunk in chunks]
            chunked_docs.extend(doc_chunks)
        doc = chunked_docs

        print(f"Total chunks: {len(doc)}")
        for i, chunk in enumerate(doc):
            prompt = chunk.text[:1500]
            try:
                await self.gemini.call(prompt)  # Can be enhanced to do graph building with LlamaIndex
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
        self.index = PropertyGraphIndex.from_documents(
            doc,
            embed_model=self.embedding_model,
            storage_context=storage_context,
            show_progress=True
        )
        return self.index

    def load_index(self, path):
        self.query_engine = self.index.as_query_engine(embed_model=self.embedding_model)
        return

    async def prediction(self, difficulty_level):
        prompt = f"""
You are an AI designed to generate multiple-choice questions (MCQs) based on a provided chapter of a book...
Provide 40 questions for {difficulty_level} level in JSON like:
{{
  "question": "...",
  "answerA": "...",
  "answerB": "...",
  "answerC": "...",
  "answerD": "...",
  "correctAnswer": "answerB"
}}
"""
        return await self.gemini.call(prompt)

    def extract_json_from_response(self, response: str):
        object_matches = re.findall(r'{[^{}]*?(?:"[^"]*":\s*"[^"]*",?)*[^{}]*?}', response, re.DOTALL)
        valid_objects = []
        for obj_str in object_matches:
            try:
                obj = json.loads(obj_str)
                valid_objects.append(obj)
            except json.JSONDecodeError:
                continue
        return valid_objects

    def add_to_json(self, json_data, difficulty_str, chapter_number):
        difficulty_map = {"easy": 1, "medium": 2, "hard": 3}
        difficulty_value = difficulty_map.get(difficulty_str.lower(), 1)
        for item in json_data:
            item["difficulty"] = difficulty_value
            item["chapterNo"] = chapter_number
        return json_data

    def clear_neo4j(self):
        self.index = None

# ------------------------------ FastAPI App ------------------------------
app = FastAPI()
graph = graphRAG()

@app.get('/')
def index():
    return {"message": "Quizaty API Optimized!"}

@app.post('/questions')
async def predict(file: Annotated[UploadFile, File()]):
    try:
        path = "./"
        graph.load_model = lambda: None  # no-op since keys/embedding initialized in constructor

        document = graph.load_doc(file, path)
        await graph.index_doc(document, path)
        graph.load_index(path)

        json_data_all = []
        for i in ["easy", "medium", "hard"]:
            response = await graph.prediction(i)
            json_data = graph.extract_json_from_response(response)
            json_data = graph.add_to_json(json_data, i, 1)
            json_data_all.extend(json_data)
            await asyncio.sleep(2)

        graph.clear_neo4j()
        return JSONResponse(content=json_data_all)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Error", "step": "general", "details": str(e)}
        )
