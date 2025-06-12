# GraphRAG - Knowledge Graph-based Question Generation System

## Overview
GraphRAG is a sophisticated question generation system that combines knowledge graphs with Retrieval-Augmented Generation (RAG) to create high-quality multiple-choice questions from various input sources. The system is built using FastAPI and leverages Neo4j for graph storage, along with advanced language models for question generation.

## System Architecture

### Core Components

1. **GraphRAG Class**
   - Main class that orchestrates the question generation process
   - Manages document processing, indexing, and question generation
   - Handles session management and cleanup

2. **Neo4j Integration**
   - Uses Neo4j as the primary graph database
   - Implements session-based database management
   - Supports dynamic database creation and cleanup

3. **Language Models**
   - Utilizes Groq's deepseek-r1-distill-llama-70b model
   - Separate models for question generation and knowledge extraction
   - Configurable embedding model (default: sentence-transformers/all-MiniLM-L6-v2)

### Key Features

1. **Session Management**
   - Isolated processing environments for each request
   - Automatic cleanup of temporary resources
   - Stale session detection and cleanup

2. **Document Processing**
   - Support for PDF files (with and without Table of Contents)
   - Text input processing
   - Chapter-based processing for structured documents

3. **Question Generation**
   - Multiple difficulty levels (easy, medium, hard)
   - Structured JSON output format
   - Batch processing for large documents

## API Endpoints

### 1. Question Generation from PDF
```http
POST /questions
```
Parameters:
- `filePDF`: PDF file upload
- `url`: Optional URL for PDF download
- `urlBool`: Boolean flag for URL processing
- `hasTOC`: Boolean flag for Table of Contents presence
- `chapters`: List of chapter numbers (when hasTOC is true)
- `chapterslndexes`: List of chapter index objects (when hasTOC is false)

### 2. Survey Questions Generation
```http
POST /survey-questions
```
Parameters:
- `filePDF`: PDF file upload

### 3. Text-based Question Generation
```http
POST /text-questions
```
Parameters:
- `text`: Input text for question generation

## Question Format

The system generates questions in the following JSON format:
```json
{
    "question": "Question text",
    "answerA": "Option A",
    "answerB": "Option B",
    "answerC": "Option C",
    "answerD": "Option D",
    "correctAnswer": "answerX",
    "difficulty": 1|2|3,
    "chapterNo": "chapter_number"
}
```

## Setup and Configuration

### Prerequisites
- Python 3.8+
- Neo4j Database
- Groq API Key

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key
- `NEO4J_URI`: Neo4j database URI
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password

### Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Configure Neo4j database
4. Set up environment variables
5. Run the server:
```bash
python final.py
```

## Error Handling

The system implements comprehensive error handling:
- Input validation
- Session cleanup on errors
- Detailed error messages
- Graceful fallback mechanisms

## Performance Considerations

1. **Resource Management**
   - Automatic cleanup of temporary files
   - Session-based isolation
   - Efficient memory usage

2. **Rate Limiting**
   - Built-in delays between batch processing
   - Configurable retry mechanisms

3. **Scalability**
   - Support for multiple concurrent sessions
   - Efficient database management
   - Optimized document processing

## Security Features

1. **Session Isolation**
   - Separate databases per session
   - Secure file handling
   - Temporary resource management

2. **Input Validation**
   - File type verification
   - URL validation
   - Parameter sanitization

## Best Practices

1. **Document Processing**
   - Use structured PDFs with Table of Contents when possible
   - Break large documents into manageable chapters
   - Validate input files before processing

2. **Question Generation**
   - Start with easy difficulty for testing
   - Monitor response quality
   - Adjust batch sizes based on document complexity

## Limitations

1. **PDF Processing**
   - Requires well-structured PDFs for optimal results
   - Limited support for complex formatting
   - Table of Contents must be properly formatted

2. **Resource Usage**
   - Memory intensive for large documents
   - Requires significant processing time
   - Neo4j database must be properly configured

## Future Improvements

1. **Planned Features**
   - Support for more document formats
   - Enhanced question quality metrics
   - Improved error recovery mechanisms

2. **Performance Optimizations**
   - Parallel processing support
   - Caching mechanisms
   - Optimized database queries

## Support and Maintenance

For issues and feature requests, please:
1. Check existing documentation
2. Review error logs
3. Contact the development team

## License

[Specify your license here] 