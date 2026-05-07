FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY src/. src/
COPY models/ models/
COPY templates/ templates/
COPY app.py .

# Create necessary directories
RUN mkdir -p data/processed data/transform logs reports/evaluation

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]