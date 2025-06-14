# Use a specific Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Expose a port if your web application (Flask/Django) listens on one
EXPOSE 5000

# Command to run your application (e.g., if you had a Flask app.py)
# This example assumes you have a Flask app file (e.g., app.py)
# CMD ["python", "app.py"]
# For now, let's just make it run your main.py for demonstration
CMD ["python", "main.py"]
