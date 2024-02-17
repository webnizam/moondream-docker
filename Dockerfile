ARG PYTHON_VERSION=3.10.10
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables to prevent Python from writing pyc files and to keep it from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache

# Set the working directory
WORKDIR /app

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create and set permissions for the cache directory
RUN mkdir -p /cache && chmod -R 777 /cache

# Copy the source code into the container
COPY . .

# Expose the port that the application listens on
EXPOSE 8000 7860

# Run the application
CMD ["python", "gradio_demo.py"]
