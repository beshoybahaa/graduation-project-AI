# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Run FastAPI server
CMD ["uvicorn", "quizzaty_api:app", "--host", "0.0.0.0", "--port", "8000"]