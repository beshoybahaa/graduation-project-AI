from typing import Annotated, List, Dict, Any, Optional
import asyncio
import json
import os
import re
import shutil
import tempfile
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    PropertyGraphIndex,
    StorageContext,
    SimpleDirectoryReader,
    Document,
    Settings
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.graph_stores.falkordb import FalkorDBGraphStore

# Configure Gemini API
genai.configure(api_key="AIzaSyAwuVnbkTAMhR5-DxwYzwBN9-vilX_bnXY")

# Response models
class Question(BaseModel):
    question: str
    answerA: str
    answerB: str
    answerC: str
    answerD: str
    correctAnswer: str
    difficulty: int
    chapterNo: int

class ErrorResponse(BaseModel):
    error: str
    step: str
    details: str

# State management
class QuizState(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    documents: Optional[List[Document]] = None
    index: Optional[PropertyGraphIndex] = None
    questions: Optional[List[Question]] = None
    error: Optional[str] = None
    current_difficulty: Optional[str] = None

class QuizzatyAPI:
    def __init__(self):
        self.storage_dir = tempfile.mkdtemp()
        self.upload_dir = tempfile.mkdtemp()
        self.llm_questions = None
        self.llm_1 = None
        self.llm_2 = None
        self.llm_3 = None
        self.embedding_model = None
        self.graph_store = None
        self._setup_graph_store()
        self._setup_models()
        self._setup_graph()

    def _setup_graph_store(self):
        try:
            self.graph_store = FalkorDBGraphStore(
                "redis://0.0.0.0:6379",
                decode_responses=True
            )
        except Exception as e:
            print(f"Warning: Could not connect to FalkorDB: {str(e)}")
            print("Falling back to in-memory graph store")
            self.graph_store = SimpleGraphStore()

    def _setup_models(self):
        # Initialize Gemini models
        self.llm_questions = Gemini(
            model="gemini-pro",
            api_key="AIzaSyAwuVnbkTAMhR5-DxwYzwBN9-vilX_bnXY",
            max_retries=2
        )
        self.llm_1 = Gemini(
            model="gemini-pro",
            api_key="AIzaSyBQfIuQshM7o4aM2t3kxC3bie67eCGG3Kk",
            max_retries=2
        )
        self.llm_2 = Gemini(
            model="gemini-pro",
            api_key="AIzaSyDgFA3k1ayTmqzuEzuFKCpGlXKko9otX6o",
            max_retries=2
        )
        self.llm_3 = Gemini(
            model="gemini-pro",
            api_key="AIzaSyBK1p3akSoS5ioEuMfuYD4Bq7K7pXqKnjw",
            max_retries=2
        )
        self.embedding_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        Settings.chunk_size = 500
        Settings.chunk_overlap = 200

    def _setup_graph(self):
        # Define the nodes
        def load_document(state: QuizState) -> QuizState:
            try:
                documents = SimpleDirectoryReader(self.upload_dir).load_data()
                state.documents = documents
                return state
            except Exception as e:
                state.error = f"Document loading error: {str(e)}"
                return state

        def process_document(state: QuizState) -> QuizState:
            if not state.documents:
                state.error = "No documents to process"
                return state

            try:
                storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
                text_splitter = TokenTextSplitter(
                    chunk_size=Settings.chunk_size,
                    chunk_overlap=Settings.chunk_overlap
                )

                chunked_docs = []
                for document in state.documents:
                    chunks = text_splitter.split_text(document.text)
                    doc_chunks = [Document(text=chunk) for chunk in chunks]
                    chunked_docs.extend(doc_chunks)

                state.documents = chunked_docs
                return state
            except Exception as e:
                state.error = f"Document processing error: {str(e)}"
                return state

        def create_index(state: QuizState) -> QuizState:
            if not state.documents:
                state.error = "No documents to index"
                return state

            try:
                storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
                state.index = PropertyGraphIndex.from_documents(
                    state.documents,
                    llm=self.llm_questions,
                    embed_model=self.embedding_model,
                    storage_context=storage_context
                )
                return state
            except Exception as e:
                state.error = f"Index creation error: {str(e)}"
                return state

        def generate_questions(state: QuizState) -> QuizState:
            if not state.index:
                state.error = "No index available for question generation"
                return state

            try:
                query_engine = state.index.as_query_engine(
                    llm=self.llm_questions,
                    embed_model=self.embedding_model
                )

                prompt = f"""Generate multiple-choice questions for {state.current_difficulty} level.
                Format each question as a JSON object with the following structure:
                {{
                    "question": "Question text",
                    "answerA": "Option A",
                    "answerB": "Option B",
                    "answerC": "Option C",
                    "answerD": "Option D",
                    "correctAnswer": "answerX"
                }}
                Generate 40 questions."""

                response = query_engine.query(prompt)
                questions = self._extract_json_from_response(str(response))
                state.questions = questions
                return state
            except Exception as e:
                state.error = f"Question generation error: {str(e)}"
                return state

        # Create the graph
        workflow = StateGraph(QuizState)

        # Add nodes
        workflow.add_node("load_document", load_document)
        workflow.add_node("process_document", process_document)
        workflow.add_node("create_index", create_index)
        workflow.add_node("generate_questions", generate_questions)

        # Add edges
        workflow.add_edge("load_document", "process_document")
        workflow.add_edge("process_document", "create_index")
        workflow.add_edge("create_index", "generate_questions")
        workflow.add_edge("generate_questions", END)

        # Set entry point
        workflow.set_entry_point("load_document")

        # Compile the graph
        self.graph = workflow.compile()

    def _extract_json_from_response(self, response: str) -> List[Dict[str, Any]]:
        object_matches = re.findall(r'{[^{}]*?(?:"[^"]*":\s*"[^"]*?",?)*[^{}]*?}', response, re.DOTALL)
        valid_objects = []
        for obj_str in object_matches:
            try:
                obj = json.loads(obj_str)
                valid_objects.append(obj)
            except json.JSONDecodeError:
                continue
        return valid_objects

    def _add_metadata_to_questions(self, questions: List[Dict[str, Any]], difficulty: str, chapter_no: int) -> List[Dict[str, Any]]:
        difficulty_map = {"easy": 1, "medium": 2, "hard": 3}
        difficulty_value = difficulty_map.get(difficulty.lower(), 1)
        
        for question in questions:
            question["difficulty"] = difficulty_value
            question["chapterNo"] = chapter_no
        
        return questions

    async def process_document(self, file: UploadFile, difficulty: str = "medium", chapter_no: int = 1) -> List[Question]:
        try:
            # Save uploaded file
            file_path = os.path.join(self.upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Initialize state
            state = QuizState(current_difficulty=difficulty)

            # Run the graph
            final_state = await self.graph.arun(state)

            if final_state.error:
                raise HTTPException(status_code=500, detail=final_state.error)

            if not final_state.questions:
                raise HTTPException(status_code=500, detail="No questions were generated")

            # Add metadata to questions
            questions = self._add_metadata_to_questions(
                final_state.questions,
                difficulty,
                chapter_no
            )

            return questions

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def __del__(self):
        for dir_path in [self.storage_dir, self.upload_dir]:
            if dir_path and os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Warning: Error cleaning up directory {dir_path}: {str(e)}")

# Create FastAPI app
app = FastAPI(title="Quizzaty API with LangGraph")
quizzaty = QuizzatyAPI()

@app.get("/")
async def index():
    return {"message": "Quizzaty API with LangGraph!"}

@app.post("/questions")
async def generate_questions(
    file: Annotated[UploadFile, File()],
    difficulty: str = "medium",
    chapter_no: int = 1
):
    try:
        questions = await quizzaty.process_document(file, difficulty, chapter_no)
        return JSONResponse(content=questions)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": str(e.detail)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"}
        ) 