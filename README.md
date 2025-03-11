# Llama Trader Agent

A scalable, AI-powered cryptocurrency trading bot that combines real-time market data collection, time series forecasting, and LLM-based decision making to execute automated trading strategies.

## Project Overview

This project implements a modular trading bot designed to run in Docker, with the following key components:

1. **Data Collection**: Fetches real-time and historical cryptocurrency data from exchanges
2. **Time Series Forecasting**: Uses the chronos-forecasting library to predict future price movements
3. **LLM Integration**: Leverages Llama 3.2 for intelligent trading decisions
4. **Automated Execution**: Handles order placement and management

## Architecture

The system consists of the following modules:

- **DataCollector (`data_collector.py`)**: Collects OHLCV, order book, and other market data using ccxt
- **AnalyticalAgent (`analytical_agent.py`)**: Performs time series forecasting using chronos-forecasting
- **Executor (`executor.py`)**: Executes buy/sell orders on cryptocurrency exchanges
- **LlamaClient (`llama_client.py`)**: Communicates with Llama 3.2 model via Ollama
- **Orchestrator (`orchestrator.py`)**: Coordinates all components and uses llama_index for data indexing
- **Main (`main.py`)**: Entry point that runs the trading cycles

## Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Ollama with Llama 3.2 model (for LLM integration)
- Exchange API credentials (optional for live trading)

## Installation

### Option 1: Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/llama-trader-agent.git
   cd llama-trader-agent
   ```

2. Create a `.env` file with your configuration (see Environment Variables section)

3. Build and run the Docker container:
   ```bash
   docker build -t llama-trader-agent .
   docker run -d --name trader --env-file .env llama-trader-agent
   ```

### Option 2: Local Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/llama-trader-agent.git
   cd llama-trader-agent
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your configuration

4. Run the bot:
   ```bash
   python main.py
   ```

## Environment Variables

The following environment variables can be set in a `.env` file or passed to the Docker container:

```
EXCHANGE_ID=binance       # Exchange to use
API_KEY=your_api_key      # Exchange API key (optional)
API_SECRET=your_secret    # Exchange API secret (optional)
TEST_MODE=True            # Use exchange sandbox/test mode
LLAMA_URL=http://localhost:11434  # Ollama API URL
LLAMA_MODEL=llama3.2      # Llama model name
SYMBOLS=BTC/USDT,ETH/USDT # Trading pairs to monitor
TIMEFRAME=1h              # Timeframe for data collection
HISTORY_LIMIT=100         # Number of historical candles to fetch
FORECAST_HORIZON=12       # Number of periods to forecast
MIN_CONFIDENCE=medium     # Minimum confidence for trade execution
TRADE_AMOUNT=0.001        # Amount to trade in base currency
CYCLE_INTERVAL=3600       # Seconds between trading cycles
LOG_LEVEL=INFO            # Logging level (DEBUG, INFO, WARNING, ERROR)
ENABLE_FILE_LOGGING=True  # Enable logging to files
```

## Running with Ollama

To use the LLM capabilities, you need to run Ollama with the Llama 3.2 model:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Llama 3.2 model:
   ```bash
   ollama pull llama3.2
   ```
3. Run Ollama:
   ```bash
   ollama serve
   ```

## Docker Compose Setup

For a complete setup with Ollama, create a `docker-compose.yml` file:

```yaml
version: '3'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    command: serve

  trader:
    build: .
    depends_on:
      - ollama
    environment:
      - LLAMA_URL=http://ollama:11434
      - TEST_MODE=True
      # Add other environment variables as needed

volumes:
  ollama-data:
```

Then run:
```bash
docker-compose up -d
```

## Logging System

The application uses a comprehensive logging system to track its operations, decisions, and interactions with the LLM model. This helps with debugging, monitoring, and auditing the trading bot's behavior.

### Log Files

The logging system creates the following log files in the `logs` directory:

- **llama_trader.log**: Main application log containing general information about the bot's operations
- **llm_interactions.log**: Detailed log of all interactions with the Llama model, including prompts sent and responses received
- **trading_activity.log**: Record of all trading activities, including order executions and their details

### Log Configuration

The logging system can be configured using environment variables:

- `LOG_LEVEL`: Sets the logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `ENABLE_FILE_LOGGING`: Enables or disables logging to files (True/False)

Example configuration in `.env` file:
```
LOG_LEVEL=DEBUG
ENABLE_FILE_LOGGING=True
```

### Log Format

Logs include timestamp, logger name, log level, and message. Debug logs also include file name, line number, and function name for easier troubleshooting:

```
2023-06-15 14:30:45 - Orchestrator - INFO - Analyzing market for BTC/USDT
2023-06-15 14:30:47 - LlamaClient - INFO - Sending prompt to llama3.2 (length: 1024 chars)
```

### Viewing Logs

To view logs in real-time while the bot is running:

```bash
# View main application log
tail -f logs/llama_trader.log

# View LLM interactions
tail -f logs/llm_interactions.log

# View trading activity
tail -f logs/trading_activity.log
```

## Usage

The bot will run continuously, performing the following steps in each cycle:

1. Collect historical data for configured trading pairs
2. Generate price forecasts using the chronos-forecasting model
3. Index the data using llama_index
4. Send a query to the Llama model for trading decisions
5. Execute trades based on the model's recommendations
6. Wait for the next cycle

## Customization

- **Trading Strategy**: Modify the prompt in `llama_client.py` to implement different trading strategies
- **Forecasting Model**: Change the model in `analytical_agent.py` to use different chronos models
- **Additional Indicators**: Extend the `analytical_agent.py` to include technical indicators
- **Logging Configuration**: Adjust the logging settings in `logger_config.py` to change log formats or rotation policies

## License

MIT License - See LICENSE file for details

## Disclaimer

This software is for educational purposes only. Use at your own risk. Cryptocurrency trading involves significant risk and may result in the loss of your invested capital.
