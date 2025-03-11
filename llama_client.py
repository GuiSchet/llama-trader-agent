#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LlamaClient Module

This module is responsible for communicating with a Llama 3.2 model running in a Docker
container through Ollama. It provides functions to send prompts to the model and receive
responses for decision-making in the trading bot.
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Union, Any
from logger_config import get_logger, log_llm_interaction

# Get logger for this module
logger = get_logger('LlamaClient')


class LlamaClient:
    """
    A class for communicating with a Llama model via Ollama API.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.2"):
        """
        Initialize the LlamaClient with the Ollama API endpoint.
        
        Args:
            base_url: The base URL of the Ollama API (default: "http://localhost:11434")
            model_name: The name of the model to use (default: "llama3.2")
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.api_url = f"{self.base_url}/api/generate"
        logger.info(f"Initialized LlamaClient with model: {model_name} at {base_url}")
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """
        Generate a response from the Llama model.
        
        Args:
            prompt: The text prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative, lower = more deterministic)
            
        Returns:
            The generated text response
        """
        try:
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            # Log the prompt details
            logger.info(f"Sending prompt to {self.model_name} (length: {len(prompt)} chars)")
            
            # Make the API request
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse the response
            result = response.json()
            generated_text = result.get('response', '')
            
            # Log the interaction with detailed information
            metadata = {
                "model": self.model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_length": len(prompt),
                "response_length": len(generated_text)
            }
            log_llm_interaction(prompt, generated_text, metadata)
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing API response: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def analyze_trading_opportunity(self, market_data: Dict, forecast_data: Dict) -> Dict:
        """
        Analyze a trading opportunity using the Llama model.
        
        Args:
            market_data: Dictionary containing market data (prices, volumes, etc.)
            forecast_data: Dictionary containing forecast data
            
        Returns:
            Dictionary containing the analysis result with decision and confidence
        """
        try:
            # Extract relevant data for the prompt
            symbol = market_data.get('symbol', 'Unknown')
            latest_prices = market_data.get('latest_prices', [])
            forecast = forecast_data.get('forecast', {})
            
            # Format the data for the prompt
            latest_prices_str = ', '.join([f"{price:.2f}" for price in latest_prices[-10:]])
            median_forecast = forecast.get('median', [])
            median_forecast_str = ', '.join([f"{price:.2f}" for price in median_forecast[:10]])
            
            # Calculate some basic metrics for the prompt
            if len(latest_prices) >= 24:
                price_change_24h = ((latest_prices[-1] / latest_prices[-24]) - 1) * 100
                price_change_str = f"{price_change_24h:.2f}%"
            else:
                price_change_str = "N/A (insufficient data)"
                
            # Calculate short-term trend (last 5 candles)
            short_term_trend = "neutral"
            if len(latest_prices) >= 5:
                if latest_prices[-1] > latest_prices[-5]:
                    short_term_trend = "bullish"
                elif latest_prices[-1] < latest_prices[-5]:
                    short_term_trend = "bearish"
            
            # Calculate forecast trend
            forecast_trend = "neutral"
            if len(median_forecast) > 0 and len(latest_prices) > 0:
                forecast_change = ((median_forecast[0] / latest_prices[-1]) - 1) * 100
                if forecast_change > 1.0:
                    forecast_trend = "bullish"
                elif forecast_change < -1.0:
                    forecast_trend = "bearish"
                forecast_change_str = f"{forecast_change:.2f}%"
            else:
                forecast_change_str = "N/A"
            
            # Create a structured prompt with more comprehensive data
            prompt = f"""
            You are a cryptocurrency trading assistant. Based on the following market data and forecasts,
            provide a trading recommendation (buy, sell, or hold) for {symbol}.
            
            MARKET DATA:
            - Symbol: {symbol}
            - Latest 10 prices: [{latest_prices_str}]
            - 24h price change: {price_change_str}
            - Short-term trend (5 periods): {short_term_trend}
            
            FORECAST DATA:
            - Next 10 periods forecast: [{median_forecast_str}]
            - Forecast trend: {forecast_trend}
            - Forecasted price change: {forecast_change_str}
            
            Analyze the trend, volatility, and forecast to make your decision. Consider both short-term and forecasted trends.
            
            Provide your recommendation in a structured format with the following fields:
            - Decision: [buy/sell/hold]
            - Confidence: [low/medium/high]
            - Reasoning: [detailed explanation with specific data points that influenced your decision]
            
            Your analysis:
            """
            
            # Get the model's response
            response = self.generate(prompt)
            
            # Parse the response to extract the decision
            decision = self._parse_trading_decision(response)
            
            # Add the full response for reference
            decision['full_response'] = response
            
            return decision
            
        except Exception as e:
            logger.error(f"Error analyzing trading opportunity: {str(e)}")
            raise
    
    def _parse_trading_decision(self, response: str) -> Dict:
        """
        Parse the model's response to extract the trading decision.
        
        Args:
            response: The raw text response from the model
            
        Returns:
            Dictionary containing the parsed decision
        """
        # Default values
        result = {
            'decision': 'hold',  # Default to hold if parsing fails
            'confidence': 'low',
            'reasoning': ''
        }
        
        try:
            # Look for decision indicators
            response_lower = response.lower()
            
            # Extract decision
            if 'decision: buy' in response_lower or 'decision:buy' in response_lower:
                result['decision'] = 'buy'
            elif 'decision: sell' in response_lower or 'decision:sell' in response_lower:
                result['decision'] = 'sell'
            elif 'decision: hold' in response_lower or 'decision:hold' in response_lower:
                result['decision'] = 'hold'
            
            # Extract confidence
            if 'confidence: high' in response_lower or 'confidence:high' in response_lower:
                result['confidence'] = 'high'
            elif 'confidence: medium' in response_lower or 'confidence:medium' in response_lower:
                result['confidence'] = 'medium'
            
            # Extract reasoning (look for the reasoning section)
            reasoning_start = response_lower.find('reasoning:')
            if reasoning_start != -1:
                reasoning_text = response[reasoning_start + 10:].strip()
                # Take only the first paragraph of reasoning
                end_idx = reasoning_text.find('\n\n')
                if end_idx != -1:
                    reasoning_text = reasoning_text[:end_idx]
                result['reasoning'] = reasoning_text
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing trading decision: {str(e)}")
            return result


# Example usage
def example():
    # Create client instance
    client = LlamaClient()
    
    try:
        # Simple generation example
        prompt = "What are the key factors to consider when trading cryptocurrencies?"
        response = client.generate(prompt)
        print(f"\nPrompt: {prompt}\n\nResponse:\n{response}\n")
        
        # Trading analysis example
        market_data = {
            'symbol': 'BTC/USDT',
            'latest_prices': [42000, 42100, 42300, 42200, 42400, 42600, 42800, 43000, 43200, 43500]
        }
        
        forecast_data = {
            'forecast': {
                'median': [43800, 44100, 44500, 44800, 45000, 45200]
            }
        }
        
        analysis = client.analyze_trading_opportunity(market_data, forecast_data)
        print(f"\nTrading Analysis:\n")
        print(f"Decision: {analysis['decision']}")
        print(f"Confidence: {analysis['confidence']}")
        print(f"Reasoning: {analysis['reasoning']}")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")


if __name__ == "__main__":
    # Run the example
    example()