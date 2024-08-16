# Use the official Python 3.11.9 image as the base image
FROM python:3.11.9


# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]