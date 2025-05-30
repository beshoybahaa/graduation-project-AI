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
        
    def __init__(self):
        self.storage_dir = tempfile.mkdtemp()
        self.upload_dir = tempfile.mkdtemp()
        
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
        
        return

    def get_or_create_graph_store(self, book_name: str, chapter_number: int):
        """Create or get an existing graph store for a specific book and chapter."""
        # Replace spaces with underscores in book name
        sanitized_book_name = book_name.replace(" ", "_")
        if "(" in sanitized_book_name:
            sanitized_book_name = sanitized_book_name.replace("(", "")
        if ")" in sanitized_book_name:
            sanitized_book_name = sanitized_book_name.replace(")", "")
        sanitized_book_name = sanitized_book_name.replace(".pdf", "")
        # Convert to camel case
        words = sanitized_book_name.split('_')
        sanitized_book_name = words[0].lower() + ''.join(word.capitalize() for word in words[1:])
        graph_name = f"{sanitized_book_name}Chapter{chapter_number}"
        
        try:
            print("Checking available Neo4j databases...")
            from neo4j import GraphDatabase
            
            # First connect to system database to list available databases
            system_driver = GraphDatabase.driver(
                "bolt://0.0.0.0:7687",
                auth=("neo4j", "mysecret")
            )
            
            with system_driver.session(database="system") as session:
                result = session.run("SHOW DATABASES")
                databases = [record["name"] for record in result]
                print(f"Available databases: {databases}")
                
                found = False

                if graph_name.lower() not in [db.lower() for db in databases]:
                    print("Creating neo4j database...")
                    session.run(f"CREATE DATABASE {graph_name}")
                    print("Database created successfully")
                    # Wait for database to be ready
                    time.sleep(5)
                    system_driver.close()
                else:
                    print("Database already exists")
                    found = True

                print("Attempting to connect to Neo4j...")
                self.base_graph_store = Neo4jPropertyGraphStore(
                    username="neo4j",
                    password="mysecret",
                    url="bolt://0.0.0.0:7687",
                    database=graph_name  # Explicitly specify the database name
                )
            if found == True:
                return found
            else:
                return self.base_graph_store
                
        except Exception as e:
            print(f"Error managing graph store: {str(e)}")
            # Fallback to in-memory graph store
            self.storage_context = SimpleGraphStore()
            return None

    def clear_neo4j(self):
        """Clear all nodes and relationships from the Neo4j database."""
        try:
            # Get the Neo4j client from the graph store and execute a clear command
            with self.graph_store._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("Successfully cleared Neo4j database")
        except Exception as e:
            print(f"Error clearing Neo4j database: {str(e)}")
            raise

    def reset_system(self):
        """Reset the entire system to ensure no old data remains."""
        try:
            # Clear the Neo4j database
            self.clear_neo4j()
            
            # Reset the index and storage context
            self.index = None
            self.storage_context = None
            
            # Clear temporary directories
            if os.path.exists(self.storage_dir):
                shutil.rmtree(self.storage_dir)
            if os.path.exists(self.upload_dir):
                shutil.rmtree(self.upload_dir)
            
            # Create fresh temporary directories
            self.storage_dir = tempfile.mkdtemp()
            self.upload_dir = tempfile.mkdtemp()
            
            print("Successfully reset the entire system")
        except Exception as e:
            print(f"Error resetting system: {str(e)}")
            raise

    def load_doc(self, file):
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

    def index_doc(self, doc):
        splitter = SentenceSplitter(
            chunk_size=500,
            chunk_overlap=150,
        )
        nodes = splitter.get_nodes_from_documents(doc)

        kg_extractor = SimpleLLMPathExtractor(
            llm=self.llm_groq,
            # max_paths_per_chunk=2,
        )

        self.index = PropertyGraphIndex(
            nodes=nodes,
            embed_model=self.embedding_model,
            kg_extractors=[kg_extractor],
            property_graph_store=self.graph_store,
            show_progress=True,
        )

        return self.index
    
    def QueryEngine(self, difficulty_level):
        query_engine = self.index.as_query_engine(
            llm=self.llm_questions,
            show_progress=True,
            storage_context=self.index.storage_context,
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

    def QueryEngine_from_existing(self, difficulty_level, storage_context):
        self.index = PropertyGraphIndex.from_existing(
                property_graph_store=storage_context,
                embed_model=self.embedding_model,
                include_embeddings=False  # Disable embeddings since Neo4j might not support vector queries
            )
        query_engine = self.index.as_query_engine(
            llm=self.llm_questions,
            show_progress=True,
            storage_context=self.index.storage_context,
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
        path = "./"
        print(f"Received file: {file.filename}")
        print(f"Chapter number: {chapter_number}")
        
        # Get or create graph store for this book/chapter
        try:
            print("Setting up graph store for book/chapter...")
            graphStore = graphrag.get_or_create_graph_store(file.filename, chapter_number)
            print("Graph store setup completed successfully")
        except Exception as e:
            print(f"Warning: Error during graph store setup: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Graph Store Setup Error", "details": str(e)}
            )

        # Reset the entire system before processing new document
        # try:
        #     print("Resetting system...")
        #     graphrag.reset_system()
        #     print("System reset completed successfully")
        # except Exception as e:
        #     print(f"Warning: Error during system reset: {str(e)}")
        #     return JSONResponse(
        #         status_code=500,
        #         content={"error": "System Reset Error", "details": str(e)}
        #     )
        if graphStore != True:
            try:
                print("Starting document loading...")
                document = graphrag.load_doc(file)
                print(f"Document loading completed. Number of documents: {len(document)}")
            except Exception as e:
                print(f"Error loading document: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "Document Loading Error", "step": "load_doc", "details": str(e)}
                )

            try:
                print("Starting document indexing...")
                graphrag.index_doc(document)
                print("Document indexing completed")
            except Exception as e:
                print(f"Error indexing document: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "Document Indexing Error", "step": "index_doc", "details": str(e)}
                )

        json_data_all = []
        for i in ["easy", "medium", "hard"]:
            try:
                print(f"Generating questions for {i} difficulty...")
                # Make multiple calls to get 40 questions total
                for batch in range(3):  # This will generate 15 questions per batch, 3 batches = 45 questions
                    if graphStore != True:
                        test = graphrag.QueryEngine(i)
                    else:
                        test = graphrag.QueryEngine_from_existing(i,graphStore)
                    response_answer = str(test)
                    json_data = graphrag.extract_json_from_response(response_answer)
                    json_data = graphrag.add_to_json(json_data, i, chapter_number)
                    json_data_all.extend(json_data)
                    time.sleep(3)  # Add delay between batches
                print(f"Completed {i} difficulty level")
            except Exception as e:
                print(f"Error generating {i} difficulty questions: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Question Generation Error for {i} difficulty",
                        "step": "QueryEngine",
                        "details": str(e)
                    }
                )
            
        print(f"Successfully generated {len(json_data_all)} total questions")
        return JSONResponse(content=json_data_all)
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Error", "step": "general", "details": str(e)}
        )

    
