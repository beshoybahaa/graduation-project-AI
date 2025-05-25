# Quizzaty API with LangGraph

A modern implementation of the Quizzaty API using LangGraph framework for better workflow management and state handling.

## Features

- Document processing and question generation using LangGraph workflow
- Multiple LLM support with Gemini models
- Graph-based document indexing with FalkorDB
- Async processing with proper error handling
- Clean API interface with FastAPI

## Requirements

- Python 3.9+
- Redis (for FalkorDB graph store)
- Google API key for Gemini models

## Installation

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

## Running the API

1. Start Redis server (required for FalkorDB):
   ```bash
   docker run -d -p 6379:6379 redis
   ```

2. Start the API server:
   ```bash
   uvicorn quizzaty_api_langgraph:app --reload
   ```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET /
Returns a welcome message.

### POST /questions
Generates multiple-choice questions from an uploaded document.

**Parameters:**
- `file`: The document file to process (PDF, TXT, etc.)
- `difficulty`: Question difficulty level ("easy", "medium", "hard")
- `chapter_no`: Chapter number for the questions

**Response:**
```json
[
  {
    "question": "Question text",
    "answerA": "Option A",
    "answerB": "Option B",
    "answerC": "Option C",
    "answerD": "Option D",
    "correctAnswer": "answerX",
    "difficulty": 1,
    "chapterNo": 1
  }
]
```

## Architecture

The API uses LangGraph to manage the document processing workflow:

1. **Document Loading**: Loads and validates the uploaded document
2. **Document Processing**: Splits the document into chunks
3. **Index Creation**: Creates a graph-based index of the document
4. **Question Generation**: Generates questions using the indexed content

Each step is managed as a node in the LangGraph workflow, with proper error handling and state management.

## Error Handling

The API includes comprehensive error handling:
- Input validation
- Document processing errors
- LLM API errors
- Graph store connection errors

All errors are returned with appropriate HTTP status codes and detailed error messages.

## License

MIT License 