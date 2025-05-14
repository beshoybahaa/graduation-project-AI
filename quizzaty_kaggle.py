import asyncio
import nest_asyncio
nest_asyncio.apply()

from typing import Union
import time
import json
import re
from math import ceil
import os
from datetime import datetime
import tempfile
import shutil
from functools import partial

# Import llama_index components
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import PropertyGraphIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.async_utils import asyncio_run

class QuizGenerator:
    def __init__(self):
        self.llm_questions = None
        self.embedding_model = None
        self.index = None
        self.query_engine = None
        self.llm_api = "YOUR_GROQ_API_KEY"  # Replace with your Groq API key
        
    def load_model(self):
        model_name_questions = "deepseek-r1-distill-llama-70b"
        self.llm_questions = Groq(model=model_name_questions, api_key=self.llm_api, max_retries=2)
        self.embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return

    def load_doc(self, pdf_path):
        documents = SimpleDirectoryReader(pdf_path).load_data()
        return documents

    async def index_doc(self, doc):
        print("Initializing shared SimpleGraphStore...")
        shared_graph_store = SimpleGraphStore()
        shared_storage_context = StorageContext.from_defaults(graph_store=shared_graph_store)
        self.index = PropertyGraphIndex.from_documents(
            doc,
            llm=self.llm_questions,
            embed_model=self.embedding_model,
            storage_context=shared_storage_context,
            show_progress=True,
            use_async=True
        )
        return self.index

    def load_index(self):
        self.query_engine = self.index.as_query_engine(
            llm=self.llm_questions,
            embed_model=self.embedding_model,
            storage_context=self.index.storage_context
        )
        return

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

    def extract_json_from_response(self, response: str):
        object_matches = re.findall(r'{[^{}]*?(?:"[^"]*":\s*"[^"]*?",?)*[^{}]*?}', response, re.DOTALL)
        valid_objects = []
        for obj_str in object_matches:
            try:
                obj = json.loads(obj_str)
                valid_objects.append(obj)
            except json.JSONDecodeError:
                continue
        return valid_objects

    def add_to_json(self, json_data, difficulty_str, chapter_number):
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

async def main():
    # Initialize the quiz generator
    quiz_generator = QuizGenerator()
    
    # Load the model
    quiz_generator.load_model()
    
    # Path to your PDF file in Kaggle
    pdf_path = "/kaggle/input/your-dataset/your-pdf-file.pdf"
    
    try:
        # Load and process the document
        document = quiz_generator.load_doc(pdf_path)
        print("Document loaded successfully")
        
        # Index the document
        await quiz_generator.index_doc(document)
        print("Document indexed successfully")
        
        # Load the index
        quiz_generator.load_index()
        print("Index loaded successfully")
        
        # Generate questions for different difficulty levels
        json_data_all = []
        for difficulty in ["easy", "medium", "hard"]:
            try:
                response = await quiz_generator.prediction(difficulty)
                response_answer = str(response)
                json_data = quiz_generator.extract_json_from_response(response_answer)
                json_data = quiz_generator.add_to_json(json_data, difficulty, 1)
                json_data_all.extend(json_data)
                print(f"Generated questions for {difficulty} difficulty")
                time.sleep(3)  # Rate limiting
            except Exception as e:
                print(f"Error generating questions for {difficulty} difficulty: {str(e)}")
        
        # Save the results to a JSON file
        output_path = "/kaggle/working/generated_questions.json"
        with open(output_path, 'w') as f:
            json.dump(json_data_all, f, indent=2)
        print(f"Questions saved to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 