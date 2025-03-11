#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AnalyticalAgent Module

This module is responsible for time series forecasting using the chronos-forecasting library.
It loads a pretrained model and provides functions to generate forecasts for cryptocurrency price data.
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
from chronos import BaseChronosPipeline, ChronosPipeline
from logger_config import get_logger

# Get logger for this module
logger = get_logger('AnalyticalAgent')


class AnalyticalAgent:
    """
    A class for time series forecasting using the chronos-forecasting library.
    """
    
    def __init__(self, model_name: str = "amazon/chronos-t5-small"):
        """
        Initialize the AnalyticalAgent with a pretrained model.
        
        Args:
            model_name: The name of the pretrained model to use
        """
        self.model_name = model_name
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def initialize(self):
        """
        Load the pretrained model.
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = BaseChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float32
            )
            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def forecast(self, time_series: Union[List[float], np.ndarray], 
                horizon: int = 12, 
                quantiles: List[float] = [0.1, 0.5, 0.9]) -> Dict[str, np.ndarray]:
        """
        Generate a forecast for the given time series.
        
        Args:
            time_series: The historical time series data (list or array of prices)
            horizon: The number of steps to forecast
            quantiles: The quantiles to predict (e.g., [0.1, 0.5, 0.9])
            
        Returns:
            Dictionary containing the forecasted quantiles
        """
        try:
            if self.model is None:
                self.initialize()
            
            # Convert input to numpy array if it's not already
            if not isinstance(time_series, np.ndarray):
                time_series = np.array(time_series)
            
            # Ensure the time series is 1D
            if time_series.ndim > 1:
                time_series = time_series.flatten()
            
            logger.info(f"Generating forecast with horizon={horizon} and quantiles={quantiles}")
            
            # Convert numpy array to torch tensor
            time_series_tensor = torch.tensor(time_series, dtype=torch.float32)
            
            # Generate forecast
            forecast_quantiles, forecast_mean = self.model.predict_quantiles(
                context=time_series_tensor,
                prediction_length=horizon,
                quantile_levels=quantiles
            )
            
            # Convert forecasts to numpy arrays
            forecast_quantiles = forecast_quantiles.numpy()
            forecast_mean = forecast_mean.numpy()
            
            logger.info(f"Successfully generated forecast with shape: {forecast_quantiles.shape}")
            
            # Create a dictionary with quantile forecasts
            result = {}
            for i, q in enumerate(quantiles):
                result[f"q{q}"] = forecast_quantiles[..., i]
            
            # Add the mean forecast
            result["mean"] = forecast_mean
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def forecast_from_dataframe(self, df: pd.DataFrame, price_column: str = 'close',
                               horizon: int = 12, 
                               quantiles: List[float] = [0.1, 0.5, 0.9]) -> Dict[str, Any]:
        """
        Generate a forecast from a DataFrame containing price data.
        
        Args:
            df: DataFrame containing the price data
            price_column: The column name containing the prices
            horizon: The number of steps to forecast
            quantiles: The quantiles to predict
            
        Returns:
            Dictionary containing the forecasted quantiles and additional metadata
        """
        try:
            # Extract the price series
            prices = df[price_column].values
            
            # Generate the forecast
            forecast = self.forecast(prices, horizon, quantiles)
            
            # Add metadata
            result = {
                "forecast": forecast,
                "last_timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None,
                "last_price": prices[-1],
                "horizon": horizon,
                "quantiles": quantiles
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating forecast from DataFrame: {str(e)}")
            raise


# Example usage
def example():
    import matplotlib.pyplot as plt
    
    # Create some sample data (simulated price series)
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 4 * np.pi, n)
    noise = np.random.normal(0, 0.5, n)
    y = np.sin(x) * 10 + 50 + noise  # Simulated price with trend and noise
    
    # Create the agent
    agent = AnalyticalAgent()
    
    try:
        # Initialize the model
        agent.initialize()
        
        # Generate forecast
        horizon = 24
        forecast = agent.forecast(y, horizon=horizon)
        
        # Plot the results
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(range(len(y)), y, label='Historical Data')
        
        # Plot forecasts
        x_forecast = range(len(y), len(y) + horizon)
        plt.plot(x_forecast, forecast['mean'], label='Mean Forecast', color='red')
        
        # Plot confidence intervals
        plt.fill_between(x_forecast, 
                         forecast['q0.1'], 
                         forecast['q0.9'], 
                         alpha=0.2, 
                         color='red', 
                         label='Confidence Interval (q=0.1 to q=0.9)')
        
        plt.title('Time Series Forecast Example')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('forecast_example.png')
        plt.close()
        
        print("Forecast example completed and saved to 'forecast_example.png'")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")


if __name__ == "__main__":
    # Run the example
    example()