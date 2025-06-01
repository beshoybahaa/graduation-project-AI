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

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, PropertyGraphIndex, StorageContext, load_index_from_storage, SimpleDirectoryReader, Document
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core import Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# from llama_index.graph_stores.falkordb import FalkorDBPropertyGraphStore
# from dotenv import load_dotenv

# Apply nest_asyncio
nest_asyncio.apply()

# Configure global LLM settings
llm = Groq(
    model="deepseek-r1-distill-llama-70b",
    api_key="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",
    max_retries=2
)
Settings.llm = llm

class graphRAG:
    embedding_model = None
    llm = None
    index = None
    query_engine = None
    graph_store = None
    storage_context = None
    active_sessions = {}  # Track active sessions
        
    def __init__(self):
        self.embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm_groq = Groq(
            model="deepseek-r1-distill-llama-70b",
            api_key="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",
            max_retries=2
        )
        self.llm_questions = Groq(
            model="deepseek-r1-distill-llama-70b",
            api_key="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",
            max_retries=2
        )

        try:
            print("Attempting to connect to Neo4j...")
            self.graph_store = Neo4jPropertyGraphStore(
                username="neo4j",
                password="mysecret",
                url="bolt://0.0.0.0:7687",
                database="neo4j"
            )
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {str(e)}")
            print("Falling back to in-memory graph store")
            self.graph_store = SimpleGraphStore()

    async def create_session(self):
        """Create a new processing session with isolated storage."""
        session = {
            'storage_dir': tempfile.mkdtemp(),
            'upload_dir': tempfile.mkdtemp(),
            'index': None,
            'storage_context': None,
            'request_id': None,
            'created_at': time.time(),
            'last_accessed': time.time()
        }
        from neo4j import GraphDatabase
        from uuid import uuid4

        request_id = str(uuid4())[:8]
        request_id = f"db{str(request_id).replace('-', '')}"

        try:
            system_driver = GraphDatabase.driver(
                    "bolt://0.0.0.0:7687",
                    auth=("neo4j", "mysecret")
                )
            with system_driver.session(database="system") as session_db:
                session_db.run(f"CREATE DATABASE {request_id}")
                time.sleep(5)  # Wait for database creation
                system_driver.close()
                session['request_id'] = request_id
                
            self.graph_store = Neo4jPropertyGraphStore(
                    username="neo4j",
                    password="mysecret",
                    url="bolt://0.0.0.0:7687",
                    database=session['request_id']
                )
            
            # Track the session
            self.active_sessions[request_id] = session
            print(f"Session created with request ID: {request_id}")
            return session
            
        except Exception as e:
            # Cleanup on failure
            if os.path.exists(session['storage_dir']):
                shutil.rmtree(session['storage_dir'])
            if os.path.exists(session['upload_dir']):
                shutil.rmtree(session['upload_dir'])
            raise Exception(f"Failed to create session: {str(e)}")

    async def cleanup_session(self, session):
        """Clean up a processing session."""
        if not session or 'request_id' not in session:
            print("Warning: Invalid session object")
            return
            
        request_id = session['request_id']
        try:
            # Clean up file system
            if os.path.exists(session['storage_dir']):
                shutil.rmtree(session['storage_dir'])
            if os.path.exists(session['upload_dir']):
                shutil.rmtree(session['upload_dir'])
                
            # Clean up database
            from neo4j import GraphDatabase
            system_driver = GraphDatabase.driver(
                    "bolt://0.0.0.0:7687",
                    auth=("neo4j", "mysecret")
                )
            with system_driver.session(database="system") as session_db:
                session_db.run(f"DROP DATABASE {request_id}")
                time.sleep(5)  # Wait for database deletion
                system_driver.close()
                
            # Remove from active sessions
            if request_id in self.active_sessions:
                del self.active_sessions[request_id]
                
            print(f"Session dropped with request ID: {request_id}")
            
        except Exception as e:
            print(f"Error cleaning up session: {str(e)}")
            # Still try to remove from active sessions
            if request_id in self.active_sessions:
                del self.active_sessions[request_id]

    async def cleanup_stale_sessions(self, max_age_seconds=3600):
        """Clean up sessions that haven't been accessed in a while."""
        current_time = time.time()
        sessions_to_cleanup = []
        
        for request_id, session in self.active_sessions.items():
            if current_time - session['last_accessed'] > max_age_seconds:
                sessions_to_cleanup.append(session)
                
        for session in sessions_to_cleanup:
            await self.cleanup_session(session)

    def update_session_access(self, session):
        """Update the last accessed time for a session."""
        if session and 'request_id' in session:
            request_id = session['request_id']
            if request_id in self.active_sessions:
                self.active_sessions[request_id]['last_accessed'] = time.time()

    async def clear_neo4j(self):
        """Clear all nodes and relationships from the Neo4j database."""
        try:
            with self.graph_store._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("Successfully cleared Neo4j database")
        except Exception as e:
            print(f"Error clearing Neo4j database: {str(e)}")
            raise

    async def load_doc(self, file, session):
        file_path = os.path.join(session['upload_dir'], file.filename)
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            documents = SimpleDirectoryReader(session['upload_dir']).load_data()
            return documents
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            raise

    async def index_doc(self, doc, session):
        splitter = SentenceSplitter(
            chunk_size=500,
            chunk_overlap=150,
        )
        nodes = splitter.get_nodes_from_documents(doc)

        kg_extractor = SimpleLLMPathExtractor(
            llm=self.llm_groq,
        )

        session['index'] = PropertyGraphIndex(
            nodes=nodes,
            embed_model=self.embedding_model,
            kg_extractors=[kg_extractor],
            property_graph_store=self.graph_store,
            show_progress=True,
        )

        return session['index']
    
    async def QueryEngine(self, difficulty_level, session):
        query_engine = session['index'].as_query_engine(
            llm=self.llm_questions,
            show_progress=True,
            storage_context=session['index'].storage_context,
            include_text=True,
        )
        response = query_engine.query(f"""You are an AI designed to generate multiple-choice questions (MCQs) based on a provided chapter of a book. Your task is to create a set of MCQs that focus on the main subject matter of the chapter. Ensure that each question is clear, concise, and relevant to the core themes of the chapter and be closed book style. Use the following structure for the MCQs:
            
            1. **Question Statement**: A clear and precise question related to the chapter content.
            2. **Answer Choices**: Four options labeled A, B, C, and D, where only one option is correct. The incorrect options should be plausible to challenge the reader's knowledge.
            3. **Correct Answer**: give me the correct answer of the question
            examples for questions : 
            1		Which of the following is not one of the components of a data communication system?
                A)	Message
                B)	Sender
                C)	Medium
                D)	All of the choices are correct

            Please ensure that the questions reflect a deep understanding of the chapter's main ideas and concepts while varying the complexity to accommodate different levels of knowledge. Provide 15 questions for {difficulty_level} level.
            
            Begin by analyzing the chapter content thoroughly to extract key concepts, terms, and themes that can be transformed into question formats. 
            
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


app = FastAPI()

# create the graphRAG object
graphrag = graphRAG()

# get request
@app.get('/')
def index():
    return {'message': 'Quizaty API!'}


# post request that takes a review (text type) and returns a sentiment score
@app.post('/questions')
async def predict(file: Annotated[UploadFile, File()], chapter_number: int = Form(1)):
    session = None
    try:
        print(f"Received file: {file.filename}")
        print(f"Chapter number: {chapter_number}")
        
        # Create a new session for this request
        session = await graphrag.create_session()
        
        try:
            print("Starting document loading...")
            document = await graphrag.load_doc(file, session)
            print(f"Document loading completed. Number of documents: {len(document)}")
        except Exception as e:
            await graphrag.cleanup_session(session)
            print(f"Error loading document: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Document Loading Error", "step": "load_doc", "details": str(e)}
            )

        try:
            print("Starting document indexing...")
            await graphrag.index_doc(document, session)
            print("Document indexing completed")
        except Exception as e:
            await graphrag.cleanup_session(session)
            print(f"Error indexing document: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Document Indexing Error", "step": "index_doc", "details": str(e)}
            )

        json_data_all = []
        for i in ["easy", "medium", "hard"]:
            try:
                print(f"Generating questions for {i} difficulty...")
                for batch in range(3):
                    test = await graphrag.QueryEngine(i, session)
                    response_answer = str(test)
                    json_data = graphrag.extract_json_from_response(response_answer)
                    json_data = graphrag.add_to_json(json_data, i, chapter_number)
                    json_data_all.extend(json_data)
                    await asyncio.sleep(3)  # Use asyncio.sleep instead of time.sleep
                print(f"Completed {i} difficulty level")
            except Exception as e:
                await graphrag.cleanup_session(session)
                print(f"Error generating {i} difficulty questions: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Question Generation Error for {i} difficulty",
                        "step": "QueryEngine",
                        "details": str(e)
                    }
                )
        
        # Clean up the session after successful processing
        await graphrag.cleanup_session(session)
            
        print(f"Successfully generated {len(json_data_all)} total questions")
        return JSONResponse(content=json_data_all)
        
    except Exception as e:
        # Ensure session cleanup in case of unexpected errors
        if session:
            await graphrag.cleanup_session(session)
        print(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Error", "step": "general", "details": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#test

    
