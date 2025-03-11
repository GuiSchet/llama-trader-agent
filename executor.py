#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Executor Module

This module is responsible for executing trades on cryptocurrency exchanges
using the ccxt library. It provides functions to place market and limit orders,
cancel orders, and check order status.
"""

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from logger_config import get_logger, log_trade_execution

# Get logger for this module
logger = get_logger('Executor')


class Executor:
    """
    A class for executing trades on cryptocurrency exchanges.
    """
    
    def __init__(self, exchange_id: str = 'binance', 
                 api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None,
                 test_mode: bool = True):
        """
        Initialize the Executor with exchange credentials.
        
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
        
        logger.info(f"Initialized Executor for exchange: {exchange_id} (test_mode: {test_mode})")
    
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
            
            # Use testnet if in test mode
            if self.test_mode and 'test' in self.exchange.urls:
                self.exchange.urls['api'] = self.exchange.urls['test']
                logger.info(f"Using test environment for {self.exchange_id}")
            
            # Load markets
            await self.exchange.load_markets()
            logger.info(f"Successfully connected to {self.exchange_id} and loaded markets")
            
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
    
    async def place_market_order(self, symbol: str, side: str, amount: float) -> Dict:
        """
        Place a market order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            
        Returns:
            Dictionary containing the order details
        """
        try:
            logger.info(f"Placing {side} market order for {amount} {symbol}")
            
            # Place market order
            order = await self.exchange.create_order(symbol, 'market', side, amount)
            
            # Log the trade execution
            price = order.get('price') or order.get('average')
            if price is None and 'trades' in order and order['trades']:
                # Calculate average price from trades if available
                trades = order['trades']
                total_cost = sum(trade.get('cost', 0) for trade in trades)
                total_amount = sum(trade.get('amount', 0) for trade in trades)
                price = total_cost / total_amount if total_amount > 0 else None
            
            # Log the trade execution with detailed information
            log_trade_execution(symbol, side, amount, price, order['id'])
            
            logger.info(f"Successfully placed {side} market order for {amount} {symbol} (Order ID: {order['id']})")
            return order
            
        except Exception as e:
            logger.error(f"Error placing {side} market order for {amount} {symbol}: {str(e)}")
            raise
    
    async def place_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Dict:
        """
        Place a limit order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: Order side ('buy' or 'sell')
            amount: Order amount in base currency
            price: Limit price
            
        Returns:
            Dictionary containing the order details
        """
        try:
            logger.info(f"Placing {side} limit order for {amount} {symbol} at price {price}")
            
            # Place limit order
            order = await self.exchange.create_order(symbol, 'limit', side, amount, price)
            
            # Log the trade execution
            log_trade_execution(symbol, side, amount, price, order['id'])
            
            logger.info(f"Successfully placed {side} limit order for {amount} {symbol} at price {price} (Order ID: {order['id']})")
            return order
            
        except Exception as e:
            logger.error(f"Error placing {side} limit order for {amount} {symbol} at price {price}: {str(e)}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """
        Cancel an order.
        
        Args:
            order_id: The ID of the order to cancel
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary containing the cancellation details
        """
        try:
            logger.info(f"Cancelling order {order_id} for {symbol}")
            
            # Cancel order
            result = await self.exchange.cancel_order(order_id, symbol)
            
            logger.info(f"Successfully cancelled order {order_id} for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id} for {symbol}: {str(e)}")
            raise
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """
        Get the status of an order.
        
        Args:
            order_id: The ID of the order
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary containing the order details
        """
        try:
            logger.info(f"Getting status for order {order_id} ({symbol})")
            
            # Get order status
            order = await self.exchange.fetch_order(order_id, symbol)
            
            logger.info(f"Order {order_id} status: {order['status']}")
            return order
            
        except Exception as e:
            logger.error(f"Error getting status for order {order_id} ({symbol}): {str(e)}")
            raise


# Example usage
async def example():
    # Create executor with test mode enabled
    executor = Executor(exchange_id='binance', test_mode=True)
    
    try:
        # Initialize
        await executor.initialize()
        
        # Place a market buy order
        symbol = 'BTC/USDT'
        side = 'buy'
        amount = 0.001  # Small amount for testing
        
        order = await executor.place_market_order(symbol, side, amount)
        print(f"Market order placed: {order}")
        
        # Close connection
        await executor.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(example())