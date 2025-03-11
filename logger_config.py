#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logger Configuration Module

This module provides a centralized logging configuration for the entire application.
It sets up console and file handlers with proper formatting and rotation policies.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

# Constants
LOG_DIRECTORY = "logs"
MAIN_LOG_FILE = "llama_trader.log"
LLM_INTERACTIONS_LOG_FILE = "llm_interactions.log"
TRADING_LOG_FILE = "trading_activity.log"

# Maximum log file size (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Number of backup files to keep
BACKUP_COUNT = 5

# Log format with timestamp, logger name, level, and message
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# More detailed format for debugging
DEBUG_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'


def setup_logging(log_level: str = "INFO", enable_file_logging: bool = True) -> None:
    """
    Set up logging configuration for the entire application.
    
    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Whether to enable logging to files
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist and file logging is enabled
    if enable_file_logging:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates when reconfiguring
    for handler in root_logger.handlers[:]:  
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    if enable_file_logging:
        # Create main log file handler with rotation
        main_log_path = os.path.join(LOG_DIRECTORY, MAIN_LOG_FILE)
        file_handler = RotatingFileHandler(
            main_log_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        )
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Create LLM interactions log file handler
        llm_log_path = os.path.join(LOG_DIRECTORY, LLM_INTERACTIONS_LOG_FILE)
        llm_handler = RotatingFileHandler(
            llm_log_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        )
        llm_handler.setLevel(numeric_level)
        llm_formatter = logging.Formatter(DEBUG_LOG_FORMAT)
        llm_handler.setFormatter(llm_formatter)
        
        # Create trading activity log file handler
        trading_log_path = os.path.join(LOG_DIRECTORY, TRADING_LOG_FILE)
        trading_handler = RotatingFileHandler(
            trading_log_path,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        )
        trading_handler.setLevel(numeric_level)
        trading_formatter = logging.Formatter(LOG_FORMAT)
        trading_handler.setFormatter(trading_formatter)
        
        # Create specific loggers for different components
        llm_logger = logging.getLogger('LlamaClient')
        llm_logger.addHandler(llm_handler)
        
        trading_logger = logging.getLogger('Executor')
        trading_logger.addHandler(trading_handler)


def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name and optional level.
    
    Args:
        name: The name of the logger
        log_level: Optional specific log level for this logger
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    
    if log_level:
        numeric_level = getattr(logging, log_level.upper(), None)
        if numeric_level:
            logger.setLevel(numeric_level)
    
    return logger


def log_llm_interaction(prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an interaction with the LLM model.
    
    Args:
        prompt: The prompt sent to the model
        response: The response received from the model
        metadata: Optional metadata about the interaction
    """
    llm_logger = logging.getLogger('LlamaClient')
    
    # Log the interaction with clear separation
    llm_logger.info("="*80)
    llm_logger.info("LLM INTERACTION")
    llm_logger.info("-"*80)
    
    # Log metadata if provided
    if metadata:
        llm_logger.info(f"Metadata: {metadata}")
    
    # Log prompt and response
    llm_logger.info(f"Prompt:\n{prompt}")
    llm_logger.info(f"Response:\n{response}")
    llm_logger.info("="*80)


def log_trade_execution(symbol: str, side: str, amount: float, price: float, order_id: str) -> None:
    """
    Log a trade execution.
    
    Args:
        symbol: The trading pair symbol
        side: Buy or sell
        amount: The amount traded
        price: The execution price
        order_id: The exchange order ID
    """
    trading_logger = logging.getLogger('Executor')
    trading_logger.info(f"TRADE EXECUTED - {symbol} - {side} - Amount: {amount} - Price: {price} - Order ID: {order_id}")