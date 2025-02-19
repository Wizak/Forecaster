# Use an official lightweight Python image.
FROM python:3.9-slim

# Install system dependencies required for building Prophet and other packages.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file and install dependencies.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port that the app runs on.
EXPOSE 5000

# Define the command to run the app.
CMD ["python", "app.py"]
