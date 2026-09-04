# Use the official Python image
FROM python:3.10-slim

# Set working directory to the project folder
WORKDIR /app

# Install system dependencies (OpenCV requires libGL)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from the retinobolic folder
COPY retinobolic/requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Retinobolic code into the container
COPY . /app

# Hugging Face Spaces automatically expose port 7860
ENV PORT=7860
EXPOSE 7860

# Set Python path to recognize custom imports
ENV PYTHONPATH="/app"

# Command to run the application (running web_app.py from within retinobolic dir)
CMD ["python", "retinobolic/web_app.py"]
