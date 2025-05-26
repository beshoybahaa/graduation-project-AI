# Quizaty API (LangChain Version with Neo4j)

This is a FastAPI-based service that processes documents and generates multiple-choice questions using LangChain and Neo4j for graph-based storage.

## Features

- Document processing and chunking
- Graph-based storage using Neo4j
- Vector storage using Neo4j Vector Search
- Concurrent processing using multiple LLMs
- Multiple-choice question generation
- Support for different difficulty levels
- JSON-formatted output

## Prerequisites

- Python 3.8+
- Neo4j Database (version 5.x)
- Google API key for Gemini model

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Neo4j Database:
   ```bash
   docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password \
     neo4j:5.14.1
   ```

5. Create the vector search index in Neo4j:
   ```cypher
   CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
   FOR (d:Document)
   ON (d.embedding)
   OPTIONS {indexConfig: {
     `vector.dimensions`: 384,
     `vector.similarity`: 'cosine'
   }}
   ```

## Running the API

Start the FastAPI server:
```bash
uvicorn quizzaty_api_langchain:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET /
- Returns a welcome message

### POST /questions
- Accepts a document file upload
- Processes the document and generates questions
- Returns JSON array of questions with metadata

## Example Usage

```python
import requests

url = "http://localhost:8000/questions"
files = {"file": open("your_document.txt", "rb")}
response = requests.post(url, files=files)
questions = response.json()
```

## Response Format

The API returns an array of question objects in the following format:

```json
{
    "question": "Question text here",
    "answerA": "Option A",
    "answerB": "Option B",
    "answerC": "Option C",
    "answerD": "Option D",
    "correctAnswer": "answerX",
    "difficulty": 1,
    "chapterNo": 1
}
```

## Error Handling

The API returns appropriate error responses with:
- HTTP status code
- Error message
- Step where the error occurred
- Detailed error information

## Notes

- The API uses 4 concurrent LLMs for faster processing
- Documents are processed in batches of 50 chunks
- Rate limiting is implemented between API calls
- Temporary files are automatically cleaned up
- Neo4j is used for both graph storage and vector search 