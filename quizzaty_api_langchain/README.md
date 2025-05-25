# Quizaty API with LangChain

This is a FastAPI-based implementation of the Quizaty API using LangChain framework for improved graph-based RAG (Retrieval-Augmented Generation) and parallel processing capabilities.

## Features

- Document processing with LangChain's document loaders
- Graph-based RAG using Neo4j
- Parallel processing with multiple LLM instances
- Vector storage using FAISS
- Question generation with difficulty levels
- FastAPI endpoints for easy integration

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
GOOGLE_API_KEY_1=your_api_key_1
GOOGLE_API_KEY_2=your_api_key_2
GOOGLE_API_KEY_3=your_api_key_3
GOOGLE_API_KEY_4=your_api_key_4
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

4. Start Neo4j (if using graph features):
Make sure Neo4j is running and accessible at the configured URL.

## Running the API

Start the FastAPI server:
```bash
uvicorn quizzaty_api:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET /
Returns a welcome message.

### POST /questions
Upload a document to generate questions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (PDF or text file)

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

## Key Improvements over Previous Version

1. **Better Graph Integration**
   - Uses Neo4j for graph storage
   - Improved graph traversal capabilities
   - Better relationship handling

2. **Enhanced Parallel Processing**
   - Uses LangChain's RunnableParallel for concurrent operations
   - Better resource management
   - More efficient batch processing

3. **Improved Document Processing**
   - Better document chunking
   - More efficient vector storage
   - Enhanced context retrieval

4. **Better Memory Management**
   - Efficient caching
   - Better context window handling
   - Improved resource utilization

## Error Handling

The API includes comprehensive error handling for:
- Document loading errors
- Processing errors
- Question generation errors
- Graph database errors

All errors return appropriate HTTP status codes and detailed error messages.

## Contributing

Feel free to submit issues and enhancement requests! 