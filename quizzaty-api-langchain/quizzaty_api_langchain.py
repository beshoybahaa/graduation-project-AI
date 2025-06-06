import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from typing import Union, Annotated, List, Dict
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import GraphQAChain
from langchain_community.graphs import Neo4jGraph
from langchain.graphs import NetworkxEntityGraph
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import nest_asyncio
from PyPDF2 import PdfReader
import networkx as nx

# Apply nest_asyncio
nest_asyncio.apply()

# Response models
class Prediction(BaseModel):
    response_answer: str

class ErrorResponse(BaseModel):
    error: str
    step: str
    details: str

class InputBody(BaseModel):
    path: str

class GraphRAG:
    def __init__(self):
        # Create temp directories
        self.storage_dir = tempfile.mkdtemp()
        self.upload_dir = tempfile.mkdtemp()
        
        # Initialize embedding model first
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize Neo4j connection
        try:
            print(f"Connecting to Neo4j at \"bolt://localhost:7687\"...")
            self.graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")



            print("Successfully connected to Neo4j graph database")
            
            # Initialize Neo4j vector store
            try:
                self.vector_store = Neo4jVector.from_existing_index(
                    embedding=self.embedding_model,
                    url="bolt://localhost:7687",
                    username="neo4j",
                    password="password",
                    index_name="document_embeddings"
                )
            except Exception as e:
                print(f"Index does not exist, creating new one: {str(e)}")
                self.vector_store = Neo4jVector(
                    embedding=self.embedding_model,
                    url="bolt://localhost:7687",
                    username="neo4j", 
                    password="password",
                    index_name="document_embeddings"
                )
            print("Successfully connected to Neo4j vector store")
        except Exception as e:
            print(f"Error connecting to Neo4j: {str(e)}")
            print("Please ensure Neo4j is running and accessible")
            print("You can start Neo4j using Docker with:")
            print("docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5.14.1")
            raise

        # Initialize LLMs
        self.llms = [
            ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                google_api_key="AIzaSyBQfIuQshM7o4aM2t3kxC3bie67eCGG3Kk",
                max_retries=2
            ),
            ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                google_api_key="AIzaSyDgFA3k1ayTmqzuEzuFKCpGlXKko9otX6o",
                max_retries=2
            ),
            ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                google_api_key="AIzaSyBK1p3akSoS5ioEuMfuYD4Bq7K7pXqKnjw",
                max_retries=2
            ),
            ChatGoogleGenerativeAI(
                model="gemma-3-27b-it",
                google_api_key="AIzaSyAwuVnbkTAMhR5-DxwYzwBN9-vilX_bnXY",
                max_retries=2
            )
        ]

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200
        )

    def __del__(self):
        # Clean up temporary directories
        for dir_path in [self.storage_dir, self.upload_dir]:
            if dir_path and os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Warning: Error cleaning up directory {dir_path}: {str(e)}")

    def load_doc(self, file: UploadFile) -> List[Document]:
        file_path = os.path.join(self.upload_dir, file.filename)
        
        try:
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Check if file is PDF
            if file.filename.lower().endswith('.pdf'):
                # Read PDF file
                pdf_reader = PdfReader(file_path)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            else:
                # Read as text file
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Convert chunks to Document objects
            documents = [Document(page_content=chunk) for chunk in chunks]
            return documents
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            raise

    async def process_chunk(self, chunk: Document, llm, chunk_index: int):
        try:
            # Add document to vector store
            self.vector_store.add_documents([chunk])
            
            # Create a new Networkx graph
            networkx_graph = NetworkxEntityGraph()
            
            # Create a graph chain for this chunk
            chain = GraphQAChain.from_llm(
                llm=llm,
                graph=networkx_graph,
                verbose=True
            )
            
            # Process the chunk and add to graph
            print(f"Processing chunk {chunk_index} with content: {chunk.page_content[:100]}...")
            result = await chain.arun(chunk.page_content)
            
            # Get the graph data
            nodes = list(networkx_graph._graph.nodes())
            edges = list(networkx_graph._graph.edges())
            
            print(f"Graph nodes after processing chunk {chunk_index}: {len(nodes)}")
            print(f"Graph edges after processing chunk {chunk_index}: {len(edges)}")
            
            # Add the processed graph to Neo4j
            for node in nodes:
                self.graph.query(
                    "MERGE (n:Entity {name: $name}) SET n += $properties",
                    {"name": node, "properties": {"type": "concept"}}
                )
            
            for edge in edges:
                self.graph.query(
                    """
                    MATCH (source:Entity {name: $source})
                    MATCH (target:Entity {name: $target})
                    MERGE (source)-[r:RELATES_TO]->(target)
                    SET r += $properties
                    """,
                    {"source": edge[0], "target": edge[1], "properties": {"type": "relationship"}}
                )
            
            return True
        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {str(e)}")
            return False

    async def index_doc(self, documents: List[Document]):
        start_time = time.time()
        chunks_per_batch = 4
        total_chunks = len(documents)
        total_batches = (total_chunks + chunks_per_batch - 1) // chunks_per_batch

        for batch_start in range(0, total_batches, len(self.llms)):
            batch_tasks = []
            
            for i, llm in enumerate(self.llms):
                batch_number = batch_start + i
                if batch_number >= total_batches:
                    continue
                
                start_idx = batch_number * chunks_per_batch
                end_idx = min(start_idx + chunks_per_batch, total_chunks)
                batch_chunks = documents[start_idx:end_idx]
                
                # Create tasks for each chunk in the batch
                chunk_tasks = [
                    self.process_chunk(chunk, llm, start_idx + j)
                    for j, chunk in enumerate(batch_chunks)
                ]
                
                batch_tasks.extend(chunk_tasks)
            
            # Wait for all chunks in this round to complete
            await asyncio.gather(*batch_tasks)
            
            # Sleep between rounds if there are more batches
            if batch_start + len(self.llms) < total_batches:
                print("\nSleeping for 30 seconds before next round...")
                await asyncio.sleep(30)
        
        end_time = time.time()
        print(f"Total indexing time: {end_time - start_time:.2f} seconds")

    async def generate_questions(self, difficulty_level: str) -> List[Dict]:
        prompt_template = """You are an AI designed to generate multiple-choice questions (MCQs) based on the knowledge graph extracted from a document. Your task is to create questions that test understanding of the relationships and concepts in the graph.

            First, analyze the knowledge graph to understand the key concepts and their relationships.
            Then, generate questions that test understanding of these concepts and relationships.
            
            For {difficulty_level} level questions:
            - Easy: Focus on basic facts and direct relationships
            - Medium: Test understanding of relationships and intermediate concepts
            - Hard: Challenge with complex relationships and deeper understanding
            
            Generate 40 questions following this structure:
            1. Question Statement: Clear and precise
            2. Four answer choices (A, B, C, D)
            3. One correct answer
            
            Format each question as a JSON object:
            {{
                "question": "Question text here",
                "answerA": "Option A",
                "answerB": "Option B",
                "answerC": "Option C",
                "answerD": "Option D",
                "correctAnswer": "answerX"  // where X is A, B, C, or D
            }}

            IMPORTANT: Your response must be a valid JSON array containing these question objects.
            Example response format:
            [
                {{
                    "question": "What is the main concept?",
                    "answerA": "Option A",
                    "answerB": "Option B",
                    "answerC": "Option C",
                    "answerD": "Option D",
                    "correctAnswer": "answerA"
                }},
                // ... more questions ...
            ]

            Use the knowledge graph to ensure questions are based on actual content from the document."""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["difficulty_level"]
        )

        # Create a new Networkx graph
        networkx_graph = NetworkxEntityGraph()

        # Use the first LLM for question generation
        chain = GraphQAChain.from_llm(
            llm=self.llms[0],
            graph=networkx_graph,
            prompt=prompt,
            verbose=True
        )

        # First, get the graph data from Neo4j
        print("Fetching graph data from Neo4j...")
        graph_data = self.graph.query("""
            MATCH (n:Entity)
            OPTIONAL MATCH (n)-[r:RELATES_TO]->(m:Entity)
            RETURN n, r, m
        """)
        
        print(f"Retrieved {len(graph_data)} records from Neo4j")
        
        # Add the data to the Networkx graph
        for record in graph_data:
            if record['n']:
                networkx_graph._graph.add_node(record['n']['name'])
            if record['m']:
                networkx_graph._graph.add_node(record['m']['name'])
            if record['r']:
                networkx_graph._graph.add_edge(record['n']['name'], record['m']['name'])

        nodes = list(networkx_graph._graph.nodes())
        edges = list(networkx_graph._graph.edges())
        print(f"Loaded graph with {len(nodes)} nodes and {len(edges)} edges")
        
        if len(nodes) == 0:
            print("Warning: No nodes found in the graph. This might affect question generation.")
            return []

        # Generate questions using the populated graph
        print(f"Generating questions for {difficulty_level} difficulty...")
        response = await chain.arun(
            query="Generate questions based on the relationships and concepts in the knowledge graph",
            difficulty_level=difficulty_level
        )
        
        print(f"Raw response from LLM: {response[:500]}...")  # Print first 500 chars of response
        
        questions = self.extract_json_from_response(response)
        print(f"Generated {len(questions)} questions for {difficulty_level} difficulty")
        
        if len(questions) == 0:
            print("Warning: No questions were generated. Raw response was:", response)
        
        return questions

    def extract_json_from_response(self, response: str) -> List[Dict]:
        try:
            # First try to parse the entire response as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            print("Failed to parse entire response as JSON, trying to extract JSON objects...")
            
            # Try to find JSON objects in the response
            object_matches = re.findall(r'{[^{}]*?(?:"[^"]*":\s*"[^"]*?",?)*[^{}]*?}', response, re.DOTALL)
            print(f"Found {len(object_matches)} potential JSON objects")
            
            valid_objects = []
            for obj_str in object_matches:
                try:
                    obj = json.loads(obj_str)
                    if all(key in obj for key in ["question", "answerA", "answerB", "answerC", "answerD", "correctAnswer"]):
                        valid_objects.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON object: {e}")
                    continue
            
            print(f"Successfully parsed {len(valid_objects)} valid question objects")
            return valid_objects

    def add_to_json(self, json_data: List[Dict], difficulty_str: str, chapter_number: int) -> List[Dict]:
        difficulty_map = {
            "easy": 1,
            "medium": 2,
            "hard": 3
        }
        
        difficulty_value = difficulty_map.get(difficulty_str.lower(), 1)
        
        for item in json_data:
            item["difficulty"] = difficulty_value
            item["chapterNo"] = chapter_number
        
        return json_data

# Create FastAPI app
app = FastAPI()
graphrag = GraphRAG()

@app.get('/')
def index():
    return {'message': 'Quizaty API (LangChain Version with Neo4j)!'}

@app.post('/questions')
async def predict(file: Annotated[UploadFile, File()]):
    try:
        # Load and process document
        try:
            documents = graphrag.load_doc(file)
            print("Document loaded successfully")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "Document Loading Error", "step": "load_doc", "details": str(e)}
            )

        # Index document
        try:
            await graphrag.index_doc(documents)
            print("Document indexed successfully")
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "Document Indexing Error", "step": "index_doc", "details": str(e)}
            )

        # Generate questions for each difficulty level
        json_data_all = []
        for difficulty in ["easy", "medium", "hard"]:
            try:
                questions = await graphrag.generate_questions(difficulty)
                questions = graphrag.add_to_json(questions, difficulty, 1)
                json_data_all.extend(questions)
                print(f"Generated questions for {difficulty} difficulty")
                await asyncio.sleep(3)  # Rate limiting
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Question Generation Error for {difficulty} difficulty",
                        "step": "generate_questions",
                        "details": str(e)
                    }
                )

        return JSONResponse(content=json_data_all)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Error", "step": "general", "details": str(e)}
        ) 
    
    #sudo docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest