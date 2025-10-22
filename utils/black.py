import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Black-76 Model Functions
def black76_call_price(F, K, T, r, sigma):
    """Calculate Black-76 call option price."""
    if T <= 0 or sigma <= 0:
        return max(F - K, 0)
    
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

def black76_put_price(F, K, T, r, sigma):
    """Calculate Black-76 put option price."""
    if T <= 0 or sigma <= 0:
        return max(K - F, 0)
    
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def black76_implied_volatility(market_price, F, K, T, r, option_type='C', max_iter=100):
    """
    Calculate implied volatility using Black-76 model.
    
    Parameters:
    - market_price: Observed market price
    - F: Forward price
    - K: Strike price
    - T: Time to expiry (in years)
    - r: Risk-free rate (use 0 for crypto)
    - option_type: 'C' for call, 'P' for put
    """
    if T <= 0:
        return np.nan
    
    price_func = black76_call_price if option_type == 'C' else black76_put_price
    
    # Intrinsic value check
    intrinsic = max(F - K, 0) if option_type == 'C' else max(K - F, 0)
    if market_price < intrinsic * np.exp(-r * T) * 0.99:  # Allow small tolerance
        return np.nan
    
    # Define objective function
    def objective(sigma):
        return price_func(F, K, T, r, sigma) - market_price
    
    try:
        # Use Brent's method to find the root
        iv = brentq(objective, 0.01, 10.0, maxiter=max_iter)
        return iv
    except (ValueError, RuntimeError):
        return np.nan