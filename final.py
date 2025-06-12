# Standard library imports
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import requests
# from datetime import datetime
from typing import Union, Annotated, Optional, List, Callable
# from math import ceil
# from functools import partial

# Third-party imports
import nest_asyncio
from fastapi import FastAPI, UploadFile, Form, File, Request
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
import PyPDF2

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

class chapterslndexesObj(BaseModel):
    number: str
    startPage: str
    endPage: str

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
            'file_dir': tempfile.mkdtemp(),
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
            if os.path.exists(session['file_dir']):
                shutil.rmtree(session['file_dir'])
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
            if os.path.exists(session['file_dir']):
                shutil.rmtree(session['file_dir'])
                
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

    async def load_doc(self, file, session, path):
        try:
            # Create a unique filename in the file_dir
            file_name = os.path.basename(path)
            file_path = os.path.join(session['file_dir'], file_name)
            
            # Write the file content to file_dir
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Read the document from file_dir
            documents = SimpleDirectoryReader(session['file_dir']).load_data()
            
            # Clean up the file after reading
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return documents
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            # Clean up file in case of error
            if os.path.exists(file_path):
                os.remove(file_path)
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
    
    def extract_toc_from_pdf(self, reader):
        """Extract the table of contents (TOC) from the PDF outline."""
        toc = []
        if reader.outline:
            for i, item in enumerate(reader.outline):
                if isinstance(item, dict):
                    title = item.get("/Title", "Untitled")
                    start_page = reader.get_destination_page_number(item) + 1  # Convert to 1-based index
                    # Calculate end_page as the start of the next section or the end of the document
                    if i + 1 < len(reader.outline):
                        next_item = reader.outline[i + 1]
                        if isinstance(next_item, dict):
                            end_page = reader.get_destination_page_number(next_item) + 1
                        else:
                            end_page = len(reader.pages)
                    else:
                        end_page = len(reader.pages)
                    toc.append({"title": title, "start_page": start_page, "end_page": end_page})
                elif isinstance(item, list):
                    # Handle nested items (subsections)
                    for sub_item in item:
                        if isinstance(sub_item, dict):
                            title = sub_item.get("/Title", "Untitled")
                            start_page = reader.get_destination_page_number(sub_item) + 1
                            toc.append({"title": title, "start_page": start_page, "end_page": start_page})
        return toc
    
    def get_subsection_range(self, toc, choice):
        """Get the start and end page of the selected subsection."""
        if 1 <= choice <= len(toc):
            selected_item = toc[choice - 1]
            next_selected_item = toc[choice]
            if "start_page" in selected_item and "end_page" in selected_item:
                print(f"the title of the selected chapter is {toc[choice]['title']}")
                return selected_item["start_page"], next_selected_item["start_page"]
            elif "children" in selected_item:
                print(f"Please select a subsection of {selected_item['title']}:")
                sub_choice = int(input("Enter the number of the subsection: "))
                return self.get_subsection_range(selected_item["children"], sub_choice)
            else:
                print("Invalid selection. Please try again.")
        else:
            print("Invalid choice. Please try again.")
        return None, None

    def extract_chapter(self, input_pdf, output_pdf, start_page, end_page):
        """Extract the specified chapter and save it to a new PDF."""
        with open(input_pdf, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            writer = PyPDF2.PdfWriter()

            # Iterate through the specified pages and add them to the writer
            for page_num in range(start_page - 1, end_page):  # Page numbers are 0-based
                writer.add_page(reader.pages[page_num])

            # Write the extracted pages to a new PDF file
            with open(output_pdf, 'wb') as output_file:
                writer.write(output_file)
        return output_pdf

class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def receive_with_logging():
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                try:
                    # Try to parse as JSON
                    body_json = json.loads(body)
                    print("\n=== Request Body ===")
                    print(json.dumps(body_json, indent=2))
                    print("===================\n")
                except:
                    # If not JSON, print raw body
                    print("\n=== Request Body ===")
                    print(body.decode() if body else "Empty body")
                    print("===================\n")
            return message

        await self.app(scope, receive_with_logging, send)

app = FastAPI()

# Add the middleware
app.add_middleware(RequestLoggingMiddleware)

# create the graphRAG object
graphrag = graphRAG()

# get request
@app.get('/')
def index():
    return {'message': 'Quizaty API!'}

def download_pdf_from_url(url, save_path):
    """Download a PDF from a URL and save it to the specified path."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"PDF downloaded and saved to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download PDF: {e}")
        return False

# post request that takes a review (text type) and returns a sentiment score
@app.post('/questions')
async def predict(
    filePDF: Annotated[UploadFile | None, File()] = None,
    url: Optional[str] = Form(None),
    urlBool: Optional[str] = Form(None),
    hasTOC: str = Form("False"),
    chapters: Optional[List[int]] = Form(None),
    chaptersIndexes: Optional[str] = Form(None)  # Changed to str to receive JSON string
):
    session = None
    try:
        # Parse chaptersIndexes from JSON string if provided
        chapters_indexes_parsed = None
        if chaptersIndexes:
            try:
                chapters_indexes_parsed = [
                    chapterslndexesObj(**chapter) 
                    for chapter in json.loads(chaptersIndexes)
                ]
            except json.JSONDecodeError as e:
                print(f"Error parsing chaptersIndexes JSON: {e}")
            except Exception as e:
                print(f"Error creating chapterslndexesObj: {e}")

        # Print the entire request body
        print("\n=== Request Body Details ===")
        print(f"filePDF: {filePDF}")
        print(f"url: {url}")
        print(f"urlBool: {urlBool}")
        print(f"hasTOC: {hasTOC}")
        print(f"chapters: {chapters}")
        print(f"chaptersIndexes (parsed): {chapters_indexes_parsed}")
        print("===========================\n")

        # Validate input parameters
        if not urlBool == "True" and filePDF is None:
            return JSONResponse(
                status_code=400,
                content={"error": "No file provided " + str(filePDF) + " " + str(urlBool)}
            )
            
        if urlBool == "True" and not url:
            return JSONResponse(
                status_code=400,
                content={"error": "urlBool is True but no URL provided"}
            )
            
        if hasTOC == "True" and not chapters:
            return JSONResponse(
                status_code=400,
                content={"error": "hasTOC is True but no chapters provided " + str(chapters)}
            )
        
        if not hasTOC == "True" and not chaptersIndexes:
            return JSONResponse(
                status_code=400,
                content={"error": "hasTOC is False but no chaptersIndexes provided \n chaptersIndexes: " + str(chaptersIndexes)}
            )

        # Create a new session for this request
        session = await graphrag.create_session()
        list_of_chapters_pdf = []
        
        if urlBool == "True":
            # Download PDF from URL
            temp_pdf_path = os.path.join(session['storage_dir'], f"{session['request_id']}_downloaded.pdf")
            if not download_pdf_from_url(url, temp_pdf_path):
                await graphrag.cleanup_session(session)
                return JSONResponse(
                    status_code=400,
                    content={"error": "Failed to download PDF from URL"}
                )
            file_path = temp_pdf_path
        else:
            # Save uploaded file
            file_path = os.path.join(session['storage_dir'], filePDF.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(filePDF.file, buffer)
        
        if hasTOC == "True":
            reader = PyPDF2.PdfReader(file_path)
            toc = graphrag.extract_toc_from_pdf(reader)
            if not toc:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Could not extract table of contents from PDF"}
                )
                
            for chapter in chapters:
                start_page, end_page = graphrag.get_subsection_range(toc, chapter)
                if start_page is None or end_page is None:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Invalid chapter range for chapter {chapter}"}
                    )
                list_of_chapters_pdf.append([chapter, graphrag.extract_chapter(file_path, f"{session['storage_dir']}/{session['request_id']}_chapter_{chapter}.pdf", start_page, end_page)])
        else:
            for chapter in chaptersIndexes:
                if not hasattr(chapter, 'number') or not hasattr(chapter, 'startPage') or not hasattr(chapter, 'endPage'):
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Invalid chapter index format"}
                    )
                list_of_chapters_pdf.append([chapter.number, graphrag.extract_chapter(file_path, f"{session['storage_dir']}/{session['request_id']}_chapter_{chapter.number}.pdf", chapter.startPage, chapter.endPage)])

        if not list_of_chapters_pdf:
            return JSONResponse(
                status_code=400,
                content={"error": "No chapters were processed"}
            )

        json_data_all = []
        for chapter in list_of_chapters_pdf:
            try:
                print("Starting document loading...")
                # Create a proper UploadFile object with file handle
                with open(chapter[1], 'rb') as f:
                    file = UploadFile(
                        filename=os.path.basename(chapter[1]),
                        file=f
                    )
                    document = await graphrag.load_doc(file, session, chapter[1])
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

            
            for difficulty in ["easy", "medium", "hard"]:
                try:
                    print(f"Generating questions for {difficulty} difficulty...")
                    for batch in range(3):  # Generate 3 batches of questions for each difficulty
                        test = await graphrag.QueryEngine(difficulty, session)
                        response_answer = str(test)
                        json_data = graphrag.extract_json_from_response(response_answer)
                        json_data = graphrag.add_to_json(json_data, difficulty, chapter[0])
                        json_data_all.extend(json_data)
                        await asyncio.sleep(3)  # Add delay between batches
                    print(f"Completed {difficulty} difficulty level")
                except Exception as e:
                    await graphrag.cleanup_session(session)
                    print(f"Error generating {difficulty} difficulty questions: {str(e)}")
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": f"Question Generation Error for {difficulty} difficulty",
                            "step": "QueryEngine",
                            "details": str(e)
                        }
                    )
        # Clean up the session after successful processing
        await graphrag.cleanup_session(session)
            
        print(f"Successfully generated {len(json_data_all)} total questions")
        print(f"Response size: {len(str(json_data_all))} bytes")
        return JSONResponse(
            content=json_data_all,
            headers={
                "Content-Type": "application/json",
                "Transfer-Encoding": "chunked"
            }
        )
            
    except Exception as e:
        # Ensure session cleanup in case of unexpected errors
        if session:
            await graphrag.cleanup_session(session)
        print(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Error", "step": "general", "details": str(e)}
        )

@app.post("/test-upload")
async def test_upload(filePDF: UploadFile = File(...)):
    return {"filename": filePDF.filename}

@app.post('/survey-questions')
async def generate_questions(filePDF: UploadFile = File(...)):
    session = None
    try:
        # Create a new session for this request
        session = await graphrag.create_session()
        
        # Save uploaded file
        file_path = os.path.join(session['storage_dir'], filePDF.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(filePDF.file, buffer)
        
        try:
            print("Starting document loading...")
            # Create a proper UploadFile object with file handle
            with open(file_path, 'rb') as f:
                file = UploadFile(
                    filename=os.path.basename(file_path),
                    file=f
                )
                document = await graphrag.load_doc(file, session, file_path)
            print(f"Document loading completed. Number of documents: {len(document)}")
        except Exception as e:
            await graphrag.cleanup_session(session)
            print(f"Error loading document: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Document Loading Error", "details": str(e)}
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
                content={"error": "Document Indexing Error", "details": str(e)}
            )

        json_data_all = []
        for difficulty in ["easy", "medium", "hard"]:
            try:
                print(f"Generating questions for {difficulty} difficulty...")
                test = await graphrag.QueryEngine(difficulty, session)
                response_answer = str(test)
                json_data = graphrag.extract_json_from_response(response_answer)
                json_data = graphrag.add_to_json(json_data, difficulty, 1)  # Using chapter 1 as default
                json_data_all.extend(json_data)
                await asyncio.sleep(3)  # Add delay between batches
                print(f"Completed {difficulty} difficulty level")
            except Exception as e:
                await graphrag.cleanup_session(session)
                print(f"Error generating {difficulty} difficulty questions: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Question Generation Error for {difficulty} difficulty",
                        "details": str(e)
                    }
                )

        # Clean up the session after successful processing
        await graphrag.cleanup_session(session)
            
        print(f"Successfully generated {len(json_data_all)} questions")
        return JSONResponse(
            content=json_data_all,
            headers={
                "Content-Type": "application/json"
            }
        )
            
    except Exception as e:
        # Ensure session cleanup in case of unexpected errors
        if session:
            await graphrag.cleanup_session(session)
        print(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Error", "details": str(e)}
        )

class TextInput(BaseModel):
    text: str

@app.post('/text-questions')
async def generate_text_questions(text_input: TextInput):
    session = None
    try:
        # Create a new session for this request
        session = await graphrag.create_session()
        
        try:
            print("Starting document processing...")
            # Create a Document from the text input
            document = [Document(text=text_input.text)]
            print("Document processing completed")
        except Exception as e:
            await graphrag.cleanup_session(session)
            print(f"Error processing text: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Text Processing Error", "details": str(e)}
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
                content={"error": "Document Indexing Error", "details": str(e)}
            )

        try:
            print("Generating easy questions...")
            test = await graphrag.QueryEngine("easy", session)
            response_answer = str(test)
            json_data = graphrag.extract_json_from_response(response_answer)
            json_data = graphrag.add_to_json(json_data, "easy", 1)  # Using chapter 1 as default
            print("Completed question generation")
        except Exception as e:
            await graphrag.cleanup_session(session)
            print(f"Error generating questions: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Question Generation Error",
                    "details": str(e)
                }
            )

        # Clean up the session after successful processing
        await graphrag.cleanup_session(session)
            
        print(f"Successfully generated {len(json_data)} questions")
        return JSONResponse(
            content=json_data,
            headers={
                "Content-Type": "application/json"
            }
        )
            
    except Exception as e:
        # Ensure session cleanup in case of unexpected errors
        if session:
            await graphrag.cleanup_session(session)
        print(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected Error", "details": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

