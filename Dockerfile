# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the virtual environment
COPY llamaenv/ /app/llamaenv/

# Copy the application code
COPY . .

# Activate virtual environment and run FastAPI server
SHELL ["/bin/bash", "-c"]
CMD source ../llamaenv/bin/activate && \
    uvicorn quizzaty_api:app --host 0.0.0.0 --port 8000