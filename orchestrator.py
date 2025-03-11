#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Orchestrator Module

This module is responsible for coordinating all the components of the trading bot.
It uses llama_index to index the outputs from the DataCollector and AnalyticalAgent,
and then formulates natural language queries to send to the LlamaClient for decision-making.
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Import custom modules
from data_collector import DataCollector
from analytical_agent import AnalyticalAgent
from executor import Executor
from llama_client import LlamaClient
from logger_config import get_logger

# Import llama_index (updated for version 0.12.23+)
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Get logger for this module
logger = get_logger('Orchestrator')


class Orchestrator:
    """
    A class for orchestrating the trading bot components.
    """
    
    def __init__(self, exchange_id: str = 'binance', 
                 api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None,
                 test_mode: bool = True,
                 llama_url: str = "http://localhost:11434",
                 llama_model: str = "llama3.2"):
        """
        Initialize the Orchestrator with all necessary components.
        
        Args:
            exchange_id: The ID of the exchange to use
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            test_mode: Whether to use the sandbox/test environment
            llama_url: The URL of the Ollama API
            llama_model: The name of the Llama model to use
        """
        # Initialize components
        self.data_collector = DataCollector(exchange_id, api_key, api_secret, test_mode)
        self.analytical_agent = AnalyticalAgent()
        self.executor = Executor(exchange_id, api_key, api_secret, test_mode)
        self.llama_client = LlamaClient(llama_url, llama_model)
        
        # Trading parameters
        self.symbols = []
        self.timeframe = '1h'
        self.history_limit = 100
        self.forecast_horizon = 12
        self.min_confidence = 'medium'  # Minimum confidence level for executing trades
        self.trade_amount = 0.001  # Default trade amount (in base currency)
        
        logger.info("Orchestrator initialized with all components")
    
    async def initialize(self):
        """
        Initialize all components.
        """
        try:
            # Initialize exchange connections
            await self.data_collector.initialize()
            await self.executor.initialize()
            
            # Initialize the analytical agent
            self.analytical_agent.initialize()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    async def close(self):
        """
        Close all connections properly.
        """
        try:
            await self.data_collector.close()
            await self.executor.close()
            logger.info("All connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")
    
    def set_trading_parameters(self, symbols: List[str], timeframe: str = '1h',
                             history_limit: int = 100, forecast_horizon: int = 12,
                             min_confidence: str = 'medium', trade_amount: float = 0.001):
        """
        Set the trading parameters.
        
        Args:
            symbols: List of trading pair symbols to monitor
            timeframe: Timeframe for the data
            history_limit: Number of historical candles to fetch
            forecast_horizon: Number of periods to forecast
            min_confidence: Minimum confidence level for executing trades
            trade_amount: Amount to trade (in base currency)
        """
        self.symbols = symbols
        self.timeframe = timeframe
        self.history_limit = history_limit
        self.forecast_horizon = forecast_horizon
        self.min_confidence = min_confidence
        self.trade_amount = trade_amount
        
        logger.info(f"Trading parameters set: {symbols}, {timeframe}, {history_limit}, {forecast_horizon}")
    
    async def analyze_market(self, symbol: str) -> Dict:
        """
        Analyze a market by collecting data, generating forecasts, and making a decision.
        
        Args:
            symbol: Trading pair symbol to analyze
            
        Returns:
            Dictionary containing the analysis results
        """
        try:
            logger.info(f"Analyzing market for {symbol}")
            
            # Step 1: Collect historical data
            ohlcv_data = await self.data_collector.fetch_ohlcv(
                symbol, 
                timeframe=self.timeframe, 
                limit=self.history_limit
            )
            
            # Step 2: Generate forecast
            forecast_data = self.analytical_agent.forecast_from_dataframe(
                ohlcv_data,
                price_column='close',
                horizon=self.forecast_horizon,
                quantiles=[0.1, 0.5, 0.9]
            )
            
            # Step 3: Index the data using llama_index
            indexed_data = self._index_market_data(symbol, ohlcv_data, forecast_data)
            
            # Step 4: Prepare market data for LLM analysis
            market_data = {
                'symbol': symbol,
                'latest_prices': ohlcv_data['close'].values.tolist(),
                'latest_volumes': ohlcv_data['volume'].values.tolist(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log detailed market data information
            logger.info(f"Market data prepared for {symbol}:")
            logger.info(f"  - Latest close prices (last 5): {[round(p, 2) for p in market_data['latest_prices'][-5:]]}")
            logger.info(f"  - Latest volumes (last 5): {[round(v, 2) for v in market_data['latest_volumes'][-5:]]}")
            logger.info(f"  - Price change (24h): {((market_data['latest_prices'][-1] / market_data['latest_prices'][-24]) - 1) * 100:.2f}%")
            logger.info(f"  - Volume change (24h): {((market_data['latest_volumes'][-1] / market_data['latest_volumes'][-24]) - 1) * 100:.2f}%")
            
            # Log forecast information
            median_forecast = forecast_data['forecast'].get('median', [])
            logger.info(f"Forecast data for {symbol}:")
            logger.info(f"  - Median forecast (next 5 periods): {[round(p, 2) for p in median_forecast[:5]]}")
            logger.info(f"  - Forecasted change: {((median_forecast[0] / market_data['latest_prices'][-1]) - 1) * 100:.2f}%")
            
            # Step 5: Get trading decision from LLM
            decision = self.llama_client.analyze_trading_opportunity(market_data, forecast_data)
            
            # Log detailed decision information
            logger.info(f"LLM trading decision for {symbol}:")
            logger.info(f"  - Decision: {decision['decision']}")
            logger.info(f"  - Confidence: {decision['confidence']}")
            logger.info(f"  - Reasoning: {decision['reasoning']}")
            
            # Combine all results
            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'ohlcv_data': ohlcv_data,
                'forecast': forecast_data,
                'decision': decision
            }
            
            logger.info(f"Analysis completed for {symbol} with decision: {decision['decision']}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {str(e)}")
            raise
    
    def _index_market_data(self, symbol: str, ohlcv_data: pd.DataFrame, 
                         forecast_data: Dict) -> VectorStoreIndex:
        """
        Index market data using llama_index for natural language querying.
        
        Args:
            symbol: Trading pair symbol
            ohlcv_data: DataFrame containing OHLCV data
            forecast_data: Dictionary containing forecast data
            
        Returns:
            VectorStoreIndex containing the indexed data
        """
        try:
            # Create a text representation of the market data
            market_text = f"Market Data for {symbol}\n\n"
            
            # Add recent price information
            market_text += "Recent Prices (last 10 periods):\n"
            recent_prices = ohlcv_data['close'].values[-10:]
            for i, price in enumerate(recent_prices):
                market_text += f"Period {i+1}: {price:.2f}\n"
            
            # Add price change information
            price_change = (recent_prices[-1] / recent_prices[0] - 1) * 100
            market_text += f"\nPrice change over last 10 periods: {price_change:.2f}%\n"
            
            # Add volume information
            recent_volumes = ohlcv_data['volume'].values[-5:]
            market_text += "\nRecent Volumes (last 5 periods):\n"
            for i, volume in enumerate(recent_volumes):
                market_text += f"Period {i+1}: {volume:.2f}\n"
            
            # Add forecast information
            market_text += "\nPrice Forecast:\n"
            median_forecast = forecast_data['forecast'].get('median', [])
            for i, price in enumerate(median_forecast):
                market_text += f"Period {i+1}: {price:.2f}\n"
            
            # Create a document from the text
            document = Document(text=market_text)
            
            # Create nodes for indexing
            nodes = [
                TextNode(text=market_text, metadata={
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                })
            ]
            
            # Configure embedding model
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Set the embedding model in Settings
            Settings.embed_model = embed_model
            
            # Create the index with the local embedding model
            index = VectorStoreIndex(nodes)
            
            logger.info(f"Successfully indexed market data for {symbol} using local embedding model")
            return index
            
        except Exception as e:
            logger.error(f"Error indexing market data: {str(e)}")
            raise
    
    async def execute_trading_decision(self, analysis_result: Dict) -> Optional[Dict]:
        """
        Execute a trading decision based on the analysis result.
        
        Args:
            analysis_result: Dictionary containing the analysis results
            
        Returns:
            Dictionary containing the order information if an order was executed, None otherwise
        """
        try:
            symbol = analysis_result['symbol']
            decision = analysis_result['decision']
            
            # Check if we should execute a trade
            if decision['decision'] == 'hold':
                logger.info(f"Decision for {symbol} is to HOLD, no action taken")
                return None
                
            # Check confidence level
            if decision['confidence'] not in ['medium', 'high']:
                logger.info(f"Confidence level {decision['confidence']} too low for {symbol}, no action taken")
                return None
            
            # Execute the order
            side = decision['decision']  # 'buy' or 'sell'
            order = await self.executor.execute_order(
                symbol=symbol,
                order_type='market',
                side=side,
                amount=self.trade_amount
            )
            
            logger.info(f"Executed {side} order for {symbol}: {order['id']}")
            return order
            
        except Exception as e:
            logger.error(f"Error executing trading decision: {str(e)}")
            raise
    
    async def run_trading_cycle(self):
        """
        Run a complete trading cycle for all configured symbols.
        """
        try:
            logger.info(f"Starting trading cycle for symbols: {self.symbols}")
            
            for symbol in self.symbols:
                # Analyze the market
                analysis = await self.analyze_market(symbol)
                
                # Execute trading decision if appropriate
                order = await self.execute_trading_decision(analysis)
                
                if order:
                    logger.info(f"Order executed for {symbol}: {order['id']}")
                else:
                    logger.info(f"No order executed for {symbol}")
                    
            logger.info("Trading cycle completed")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            raise


# Example usage
async def example():
    # Create orchestrator instance
    orchestrator = Orchestrator(test_mode=True)
    
    try:
        # Initialize all components
        await orchestrator.initialize()
        
        # Set trading parameters
        orchestrator.set_trading_parameters(
            symbols=['BTC/USDT', 'ETH/USDT'],
            timeframe='1h',
            history_limit=100,
            forecast_horizon=12,
            min_confidence='medium',
            trade_amount=0.001
        )
        
        # Run a single trading cycle
        await orchestrator.run_trading_cycle()
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
    finally:
        # Always close connections properly
        await orchestrator.close()


if __name__ == "__main__":
    # Run the example
    asyncio.run(example())