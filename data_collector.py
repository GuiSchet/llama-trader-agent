#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DataCollector Module

This module is responsible for collecting market data from cryptocurrency exchanges
using the ccxt library. It provides functions to fetch historical OHLCV data,
order book data, and other market information.
"""

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from logger_config import get_logger

# Get logger for this module
logger = get_logger('DataCollector')


class DataCollector:
    """
    A class for collecting market data from cryptocurrency exchanges.
    """
    
    def __init__(self, exchange_id: str = 'binance', 
                 api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None,
                 test_mode: bool = True):
        """
        Initialize the DataCollector with exchange credentials.
        
        Args:
            exchange_id: The ID of the exchange to use
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            test_mode: Whether to use the sandbox/test environment
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.test_mode = test_mode
        self.exchange = None
        
        logger.info(f"Initialized DataCollector for exchange: {exchange_id} (test_mode: {test_mode})")
    
    async def initialize(self):
        """
        Initialize the exchange connection.
        """
        try:
            # Get the exchange class
            exchange_class = getattr(ccxt_async, self.exchange_id)
            
            # Create exchange instance
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
            })
            
            # For Binance, set some options to ensure compatibility
            if self.exchange_id.lower() == 'binance':
                # Extend receive window to avoid timestamp errors
                self.exchange.options['recvWindow'] = 60000
            
            try:
                # Load markets
                await self.exchange.load_markets()
                logger.info(f"Successfully connected to {self.exchange_id} and loaded markets")
            except Exception as e:
                # Re-raise the exception
                raise
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange connection: {str(e)}")
            raise
    
    async def close(self):
        """
        Close the exchange connection.
        """
        if self.exchange is not None:
            await self.exchange.close()
            logger.info(f"Closed connection to {self.exchange_id}")
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch historical OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for the data (e.g., '1m', '5m', '1h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame containing the OHLCV data
        """
        try:
            logger.info(f"Fetching OHLCV data for {symbol} ({timeframe}, limit={limit})")
            
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Successfully fetched {len(df)} OHLCV records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {str(e)}")
            raise
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        Fetch order book data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Number of orders to fetch on each side
            
        Returns:
            Dictionary containing the order book data
        """
        try:
            logger.info(f"Fetching order book for {symbol} (limit={limit})")
            
            # Fetch order book
            order_book = await self.exchange.fetch_order_book(symbol, limit)
            
            logger.info(f"Successfully fetched order book for {symbol} with {len(order_book['bids'])} bids and {len(order_book['asks'])} asks")
            return order_book
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            raise
    
    async def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch ticker data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary containing the ticker data
        """
        try:
            logger.info(f"Fetching ticker for {symbol}")
            
            # Fetch ticker
            ticker = await self.exchange.fetch_ticker(symbol)
            
            logger.info(f"Successfully fetched ticker for {symbol} (last price: {ticker['last']})")
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            raise
    
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Fetch recent trades.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Number of trades to fetch
            
        Returns:
            List of dictionaries containing trade data
        """
        try:
            logger.info(f"Fetching trades for {symbol} (limit={limit})")
            
            # Fetch trades
            trades = await self.exchange.fetch_trades(symbol, limit=limit)
            
            logger.info(f"Successfully fetched {len(trades)} trades for {symbol}")
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {str(e)}")
            raise


# Example usage
async def example():
    # Create data collector
    collector = DataCollector(exchange_id='binance', test_mode=True)
    
    try:
        # Initialize
        await collector.initialize()
        
        # Fetch OHLCV data
        ohlcv_data = await collector.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=24)
        print(f"OHLCV data:\n{ohlcv_data.head()}")
        
        # Fetch order book
        order_book = await collector.fetch_order_book('BTC/USDT', limit=5)
        print(f"Order book:\n{order_book}")
        
        # Close connection
        await collector.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(example())