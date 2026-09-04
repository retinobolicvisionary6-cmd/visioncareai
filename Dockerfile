# Use the official Python image
FROM python:3.10-slim

# Set working directory to the project folder
WORKDIR /app

# Install system dependencies (Skipping libGL since we use opencv-python-headless)

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
