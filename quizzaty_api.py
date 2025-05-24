# Standard library imports
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
# from datetime import datetime
from typing import Union, Annotated, Any
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
from llama_index.graph_stores.falkordb import FalkorDBGraphStore
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core import Document, ServiceContext
import concurrent.futures


from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SimpleLLMPathExtractor,
)
from llama_index.core import Settings

from llama_index.core.workflow import step, Context, Workflow, Event, StartEvent, StopEvent

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
        # api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCnhkm10JspaX-SPOw8eCtDeYsu8l52fiA")
        # genai.configure(api_key="AIzaSyCnhkm10JspaX-SPOw8eCtDeYsu8l52fiA")
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

    def process_batch(self, batch, llm, llm_name, i, chunk_start_time):
        batch_triplets = []
    
        for chunk in batch:
            try:
                # Create a path extractor for each chunk
                path_extractor = SimpleLLMPathExtractor(
                    llm=llm,
                    max_paths_per_chunk=20
                )
                
                # Create a node from the chunk's text content
                node = Document(text=str(chunk.text))
                
                # Extract paths from the node
                paths = path_extractor.extract_paths([node])
                
                # Convert paths to triplets
                for path in paths:
                    if len(path) >= 3:
                        subject = path[0]
                        predicate = path[1]
                        obj = path[2]
                        batch_triplets.append((subject, predicate, obj))
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
                
        chunk_end_time = time.time()
        chunk_duration = chunk_end_time - chunk_start_time
        print(f"Processed {llm_name} batch number {i} in {chunk_duration:.2f} seconds")
        return batch_triplets

    async def index_doc(self, doc, path):
        print("Initializing shared SimpleGraphStore...")
        # Create storage context with the graph store
        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        
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
        # Initialize text splitter with specified chunk size and overlap
        text_splitter = TokenTextSplitter(
            chunk_size=Settings.chunk_size,
            chunk_overlap=Settings.chunk_overlap
        )
        
        # Process each document and split into chunks
        chunked_docs = []
        for document in doc:
            chunks = text_splitter.split_text(document.text)
            # Convert chunks back to Document objects
            doc_chunks = [Document(text=chunk) for chunk in chunks]
            chunked_docs.extend(doc_chunks)
            
        # Replace original documents with chunked version
        doc = chunked_docs
        print(f"doc : {len(doc)}")
        
        # Start timer
        start_time = time.time()

        # Prepare LLM batches
        llms = [
            [self.llm_questions, "llm_questions"],
            [self.llm_1, "llm_1"],
            [self.llm_2, "llm_2"],
            [self.llm_3, "llm_3"],
        ]

        batch_size = len(doc) // 4 + 1
        batches = [
            doc[i:i + batch_size] 
            for i in range(0, len(doc), batch_size)
        ]

        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, batch in enumerate(batches):
                llm, llm_name = llms[i % 4]
                future = executor.submit(
                    self.process_batch,
                    batch,
                    llm,
                    llm_name,
                    i,
                    start_time
                )
                futures.append(future)
            
            all_triplets = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_result = future.result()
                    all_triplets.extend(batch_result)
                except Exception as exc:
                    print(f"Batch processing failed: {exc}")

        graph_store = SimplePropertyGraphStore()
        for triplet in all_triplets:
            graph_store.upsert_triplet(*triplet)

        # Create the Property Graph Index
        service_context = ServiceContext.from_defaults(
            llm=llms[0][0]  # Use first instance as default for queries
        )
        self.index = PropertyGraphIndex.from_graph_store(
            graph_store=graph_store,
            service_context=service_context,
            storage_context = storage_context
        )

        print("All tasks completed")
        total_end_time = time.time()
        total_duration = total_end_time - start_time
        print(f"Total time taken: {total_duration:.2f} seconds")
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
class DocumentProcessEvent(Event):
    document: Document
    llm: Any
    llm_name: str

class DocumentResultEvent(Event):
    index: Any
    llm_name: str

class ParallelIndexWorkflow(Workflow):
    def __init__(self, embedding_model, storage_context):
        super().__init__()  # Initialize the parent Workflow class
        self.embedding_model = embedding_model
        self.storage_context = storage_context
        self.index = None

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> DocumentProcessEvent:
        # Get the documents and LLMs from context
        doc = await ctx.get("doc")
        llms = await ctx.get("llms")
        
        # Send events for each document and LLM combination
        for i, (batch, llm, llm_name) in enumerate(llms):
            for single_doc in batch:
                ctx.send_event(DocumentProcessEvent(
                    document=single_doc,
                    llm=llm,
                    llm_name=llm_name
                ))
        return None

    @step(num_workers=4)  # Process 4 documents in parallel
    async def process_document(self, ev: DocumentProcessEvent) -> DocumentResultEvent:
        index = await asyncio.to_thread(
            PropertyGraphIndex.from_documents,
            [ev.document],
            llm=ev.llm,
            embed_model=self.embedding_model,
            storage_context=self.storage_context,
        )
        return DocumentResultEvent(index=index, llm_name=ev.llm_name)

    @step
    async def combine_results(self, ctx: Context, ev: DocumentResultEvent) -> StopEvent | None:
        # Collect all results
        results = ctx.collect_events(ev, [DocumentResultEvent] * len(await ctx.get("doc")))
        if results is None:
            return None

        # Combine all indices
        self.index = results[0].index  # Use the first index as base
        for result in results[1:]:
            # Merge the indices
            self.index.merge(result.index)
        
        return StopEvent(result=self.index)



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
