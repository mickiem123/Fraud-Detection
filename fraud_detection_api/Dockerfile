# Dockerfile

# --- STAGE 1: DEFINE THE BASE ENVIRONMENT ---
FROM python:3.11-slim

# Set environment variables for best practices in a containerized Python environment.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container.
WORKDIR /app

# --- STAGE 2: INSTALL DEPENDENCIES ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- STAGE 3: COPY THE APPLICATION CODE AND ARTIFACTS ---
COPY . .

# --- STAGE 4: DEFINE RUNTIME CONFIGURATION ---
EXPOSE 80
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:80", "main:app"]
