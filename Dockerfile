# Use official Python 3.12 image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run main.py by default
CMD ["python", "main.py"]
