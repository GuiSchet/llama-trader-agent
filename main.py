#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Module

This is the main entry point for the trading bot application.
It instantiates the Orchestrator and runs the complete trading process.
"""

import asyncio
import os
import logging
import platform
from dotenv import load_dotenv
from orchestrator import Orchestrator
from logger_config import setup_logging, get_logger

# Configure Windows-specific event loop policy to fix asyncio compatibility issues
if platform.system() == 'Windows':
    import asyncio.windows_events
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Set up logging with file rotation
setup_logging(log_level="INFO", enable_file_logging=True)
logger = get_logger('Main')

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment variables
EXCHANGE_ID = os.getenv('EXCHANGE_ID', 'binance')
API_KEY = os.getenv('API_KEY', None)
API_SECRET = os.getenv('API_SECRET', None)
TEST_MODE = os.getenv('TEST_MODE', 'False').lower() in ('true', '1', 't')

# Llama model settings
LLAMA_URL = os.getenv('LLAMA_URL', 'http://localhost:11434')
LLAMA_MODEL = os.getenv('LLAMA_MODEL', 'llama3.2')

# Trading parameters
SYMBOLS = os.getenv('SYMBOLS', 'BTC/USDT,ETH/USDT').split(',')
TIMEFRAME = os.getenv('TIMEFRAME', '1h')
HISTORY_LIMIT = int(os.getenv('HISTORY_LIMIT', '100'))
FORECAST_HORIZON = int(os.getenv('FORECAST_HORIZON', '12'))
MIN_CONFIDENCE = os.getenv('MIN_CONFIDENCE', 'medium')
TRADE_AMOUNT = float(os.getenv('TRADE_AMOUNT', '0.001'))
CYCLE_INTERVAL = int(os.getenv('CYCLE_INTERVAL', '3600'))  # Default: 1 hour


async def main():
    """
    Main function to run the trading bot.
    """
    logger.info("Starting trading bot")
    
    # Create the orchestrator
    orchestrator = Orchestrator(
        exchange_id=EXCHANGE_ID,
        api_key=API_KEY,
        api_secret=API_SECRET,
        test_mode=TEST_MODE,
        llama_url=LLAMA_URL,
        llama_model=LLAMA_MODEL
    )
    
    try:
        # Initialize all components
        await orchestrator.initialize()
        logger.info("Orchestrator initialized")
        
        # Set trading parameters
        orchestrator.set_trading_parameters(
            symbols=SYMBOLS,
            timeframe=TIMEFRAME,
            history_limit=HISTORY_LIMIT,
            forecast_horizon=FORECAST_HORIZON,
            min_confidence=MIN_CONFIDENCE,
            trade_amount=TRADE_AMOUNT
        )
        logger.info(f"Trading parameters set: {SYMBOLS}, {TIMEFRAME}, {HISTORY_LIMIT}, {FORECAST_HORIZON}")
        
        # Run trading cycles continuously
        while True:
            try:
                logger.info("Starting trading cycle")
                await orchestrator.run_trading_cycle()
                logger.info(f"Trading cycle completed. Waiting {CYCLE_INTERVAL} seconds until next cycle")
                
                # Wait for the next cycle
                await asyncio.sleep(CYCLE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {str(e)}")
                # Wait a bit before retrying
                await asyncio.sleep(60)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
    finally:
        # Clean up resources
        await orchestrator.close()
        logger.info("Trading bot shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())