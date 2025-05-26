# Quizaty API (LangChain Version)

This is a FastAPI-based service that processes documents and generates multiple-choice questions using LangChain and FalkorDB for graph-based storage.

## Features

- Document processing and chunking
- Graph-based storage using FalkorDB
- Concurrent processing using multiple LLMs
- Multiple-choice question generation
- Support for different difficulty levels
- JSON-formatted output

## Prerequisites

- Python 3.8+
- Docker (for FalkorDB)
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
4. Start FalkorDB using Docker:
   ```bash
   docker run -d -p 6379:6379 falkordb/falkordb:latest
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