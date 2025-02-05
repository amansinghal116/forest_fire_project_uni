# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt ./

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY app/ ./app

# Expose the port the app runs on (change if your app uses a different port)
EXPOSE 8000

# Define environment variables to run the app
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]