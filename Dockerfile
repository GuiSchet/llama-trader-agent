# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code into the container
COPY . .

# Create a .env file with default values if not provided
RUN echo "# Default configuration - override these values with environment variables\n\
EXCHANGE_ID=binance\n\
TEST_MODE=True\n\
LLAMA_URL=http://ollama:11434\n\
LLAMA_MODEL=llama3.2\n\
SYMBOLS=BTC/USDT,ETH/USDT\n\
TIMEFRAME=1h\n\
HISTORY_LIMIT=100\n\
FORECAST_HORIZON=12\n\
MIN_CONFIDENCE=medium\n\
TRADE_AMOUNT=0.001\n\
CYCLE_INTERVAL=3600\n\
" > .env

# Run the bot when the container launches
CMD ["python", "main.py"]