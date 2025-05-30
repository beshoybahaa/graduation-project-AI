# Standard library imports
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import uuid
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

    async def create_session(self):
        """Create a new processing session with isolated storage and temporary Neo4j database."""
        session_id = str(uuid.uuid4())
        session = {
            'storage_dir': tempfile.mkdtemp(),
            'upload_dir': tempfile.mkdtemp(),
            'index': None,
            'storage_context': None,
            'session_id': session_id,
            'graph_store': None
        }
        
        try:
            # First create a connection to the system database
            system_store = Neo4jPropertyGraphStore(
                username="neo4j",
                password="mysecret",
                url="bolt://0.0.0.0:7687",
                database="system"  # Connect to system database first
            )
            
            # Create the temporary database using the correct procedure call
            with system_store._driver.session() as neo4j_session:
                # Create the database
                neo4j_session.run("CALL dbms.createDatabase($dbName)", {"dbName": f"temp_{session_id}"})
                # Wait for the database to be ready
                neo4j_session.run("CALL dbms.waitForDatabase($dbName)", {"dbName": f"temp_{session_id}"})
            
            # Close the system connection
            system_store._driver.close()
            
            # Now connect to the newly created database
            session['graph_store'] = Neo4jPropertyGraphStore(
                username="neo4j",
                password="mysecret",
                url="bolt://0.0.0.0:7687",
                database=f"temp_{session_id}"
            )
            
            # Verify the connection
            with session['graph_store']._driver.session() as neo4j_session:
                neo4j_session.run("RETURN 1")
                
        except Exception as e:
            print(f"Warning: Could not create temporary Neo4j database: {str(e)}")
            print("Falling back to in-memory graph store")
            session['graph_store'] = SimpleGraphStore()
            
        return session

    async def cleanup_session(self, session):
        """Clean up a processing session and its temporary Neo4j database."""
        try:
            if os.path.exists(session['storage_dir']):
                shutil.rmtree(session['storage_dir'])
            if os.path.exists(session['upload_dir']):
                shutil.rmtree(session['upload_dir'])
                
            # Clean up temporary Neo4j database if it exists
            if session['graph_store'] and isinstance(session['graph_store'], Neo4jPropertyGraphStore):
                try:
                    # Connect to system database to drop the temporary database
                    system_store = Neo4jPropertyGraphStore(
                        username="neo4j",
                        password="mysecret",
                        url="bolt://0.0.0.0:7687",
                        database="system"
                    )
                    
                    with system_store._driver.session() as neo4j_session:
                        # Drop the database using the correct procedure call
                        neo4j_session.run("CALL dbms.dropDatabase($dbName)", {"dbName": f"temp_{session['session_id']}"})
                    
                    # Close both connections
                    system_store._driver.close()
                    session['graph_store']._driver.close()
                except Exception as e:
                    print(f"Error cleaning up Neo4j database: {str(e)}")
                    
        except Exception as e:
            print(f"Error cleaning up session: {str(e)}")

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
            property_graph_store=session['graph_store'],
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
        print(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Error", "step": "general", "details": str(e)}
        )

    
