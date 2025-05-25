# Standard library imports
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from typing import Union, Annotated, List, Dict, Any

# Third-party imports
import nest_asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai

# LangChain imports
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import GraphQAChain
from langchain_community.llms import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.graphs import FalkorDBGraph
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

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

# GraphRAG class using LangChain
class GraphRAG:
    def __init__(self):
        # Create temp directories
        self.storage_dir = tempfile.mkdtemp()
        self.upload_dir = tempfile.mkdtemp()
        
        # Initialize LLMs
        self.llm_1 = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key="AIzaSyAwuVnbkTAMhR5-DxwYzwBN9-vilX_bnXY",
            temperature=0.7
        )
        self.llm_2 = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key="AIzaSyBQfIuQshM7o4aM2t3kxC3bie67eCGG3Kk",
            temperature=0.7
        )
        self.llm_3 = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key="AIzaSyDgFA3k1ayTmqzuEzuFKCpGlXKko9otX6o",
            temperature=0.7
        )
        self.llm_questions = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key="AIzaSyBK1p3akSoS5ioEuMfuYD4Bq7K7pXqKnjw",
            temperature=0.7
        )
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200
        )
        
        # Initialize FalkorDB graph store
        try:
            self.graph = FalkorDBGraph(
                url="redis://0.0.0.0:6379",
                decode_responses=True
            )
        except Exception as e:
            print(f"Warning: Could not connect to FalkorDB: {str(e)}")
            self.graph = None

    def __del__(self):
        # Clean up temporary directories
        for dir_path in [self.storage_dir, self.upload_dir]:
            if dir_path and os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Warning: Error cleaning up directory {dir_path}: {str(e)}")

    async def load_document(self, file: UploadFile) -> List[Document]:
        """Load and process the uploaded document."""
        try:
            file_path = os.path.join(self.upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Load document based on file type
            if file.filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                documents = [Document(page_content=text)]
            
            return documents
        except Exception as e:
            raise Exception(f"Error loading document: {str(e)}")

    async def process_documents(self, documents: List[Document]) -> None:
        """Process documents and create graph structure."""
        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Create parallel processing chain for graph creation
            parallel_chain = RunnableParallel(
                {
                    "chunks": RunnablePassthrough(),
                    "llm": self.llm_questions
                }
            )
            
            # Process chunks and create graph nodes
            for chunk in chunks:
                # Create nodes for each chunk
                node_id = f"chunk_{hash(chunk.page_content)}"
                self.graph.add_node(
                    node_id,
                    properties={
                        "content": chunk.page_content,
                        "type": "chunk"
                    }
                )
                
                # Extract key concepts and create relationships
                concepts = await self.llm_questions.agenerate(
                    [f"Extract key concepts from this text and return as a list: {chunk.page_content}"]
                )
                
                for concept in concepts.generations[0][0].text.split('\n'):
                    if concept.strip():
                        concept_id = f"concept_{hash(concept)}"
                        self.graph.add_node(
                            concept_id,
                            properties={
                                "name": concept.strip(),
                                "type": "concept"
                            }
                        )
                        self.graph.add_edge(
                            node_id,
                            concept_id,
                            properties={"type": "contains"}
                        )
            
            # Create question generation chain
            question_prompt = PromptTemplate(
                template="""You are an AI designed to generate multiple-choice questions (MCQs) based on the provided context. 
                Create questions that are clear, concise, and relevant to the core themes. Use this structure:
                
                Question: [Your question here]
                A) [Option A]
                B) [Option B]
                C) [Option C]
                D) [Option D]
                Correct Answer: [Letter of correct answer]
                
                Context: {context}
                Difficulty Level: {difficulty}
                
                Generate 40 questions in JSON format like this:
                {{
                    "question": "Question text",
                    "answerA": "Option A",
                    "answerB": "Option B",
                    "answerC": "Option C",
                    "answerD": "Option D",
                    "correctAnswer": "answerX"
                }}""",
                input_variables=["context", "difficulty"]
            )
            
            self.question_chain = LLMChain(
                llm=self.llm_questions,
                prompt=question_prompt
            )
            
            # Create graph QA chain
            if self.graph:
                self.graph_chain = GraphQAChain.from_llm(
                    llm=self.llm_questions,
                    graph=self.graph,
                    verbose=True
                )
            
        except Exception as e:
            raise Exception(f"Error processing documents: {str(e)}")

    async def generate_questions(self, difficulty_level: str) -> List[Dict[str, Any]]:
        """Generate questions based on the graph structure."""
        try:
            # Get context from graph
            if self.graph:
                # Query the graph for relevant concepts
                concepts = self.graph.query(
                    "MATCH (c:concept) RETURN c.name as concept"
                )
                
                # Use concepts to generate questions
                context = "\n".join([c["concept"] for c in concepts])
            else:
                context = "No graph context available"
            
            # Generate questions
            response = await self.question_chain.arun(
                context=context,
                difficulty=difficulty_level
            )
            
            # Extract JSON from response
            questions = self.extract_json_from_response(response)
            
            # Add difficulty and chapter number
            questions = self.add_to_json(questions, difficulty_level, 1)
            
            return questions
            
        except Exception as e:
            raise Exception(f"Error generating questions: {str(e)}")

    def extract_json_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract JSON objects from the response string."""
        object_matches = re.findall(r'{[^{}]*?(?:"[^"]*":\s*"[^"]*?",?)*[^{}]*?}', response, re.DOTALL)
        
        valid_objects = []
        for obj_str in object_matches:
            try:
                obj = json.loads(obj_str)
                valid_objects.append(obj)
            except json.JSONDecodeError:
                continue
        
        return valid_objects

    def add_to_json(self, json_data: List[Dict[str, Any]], difficulty_str: str, chapter_number: int) -> List[Dict[str, Any]]:
        """Add difficulty and chapter number to each question."""
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

    def clear_graph(self):
        """Clear the graph database."""
        if self.graph:
            self.graph.query("MATCH (n) DETACH DELETE n")

# Create FastAPI app
app = FastAPI()

# Create GraphRAG instance
graphrag = GraphRAG()

@app.get('/')
def index():
    return {'message': 'Quizaty API with LangChain and FalkorDB!'}

@app.post('/questions')
async def predict(file: Annotated[UploadFile, File()]):
    try:
        # Load and process document
        documents = await graphrag.load_document(file)
        await graphrag.process_documents(documents)
        
        # Generate questions for each difficulty level
        json_data_all = []
        for difficulty in ["easy", "medium", "hard"]:
            try:
                questions = await graphrag.generate_questions(difficulty)
                json_data_all.extend(questions)
                await asyncio.sleep(3)  # Rate limiting
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Question Generation Error for {difficulty} difficulty",
                        "step": "prediction",
                        "details": str(e)
                    }
                )
        
        # Cleanup
        try:
            graphrag.clear_graph()
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")
        
        return JSONResponse(content=json_data_all)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Error", "step": "general", "details": str(e)}
        ) 