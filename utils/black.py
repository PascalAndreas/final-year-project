import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Black-76 Model Functions (Vectorized)
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


def black76_vega(F, K, T, r, sigma):
    """
    Calculate vega (derivative of price w.r.t. volatility) for Black-76 model.
    Vega is the same for calls and puts.
    
    Parameters are vectorized numpy arrays or scalars.
    """
    # Handle edge cases
    mask = (T > 0) & (sigma > 0) & (F > 0) & (K > 0)
    
    vega = np.zeros_like(F, dtype=float)
    
    if np.any(mask):
        d1 = (np.log(F[mask] / K[mask]) + 0.5 * sigma[mask]**2 * T[mask]) / (sigma[mask] * np.sqrt(T[mask]))
        vega[mask] = F[mask] * np.exp(-r * T[mask]) * norm.pdf(d1) * np.sqrt(T[mask])
    
    return vega


def black76_price_vectorized(F, K, T, r, sigma, option_type):
    """
    Vectorized Black-76 pricing for calls and puts.
    
    Parameters:
    - F, K, T, r, sigma: numpy arrays or scalars
    - option_type: numpy array of 'C' or 'P' strings
    
    Returns: numpy array of prices
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    option_type = np.asarray(option_type)
    
    # Initialize output
    prices = np.zeros_like(F, dtype=float)
    
    # Compute valid mask
    valid = (T > 0) & (sigma > 0) & (F > 0) & (K > 0)
    
    if not np.any(valid):
        return prices
    
    # Compute d1 and d2
    d1 = np.full_like(F, np.nan)
    d2 = np.full_like(F, np.nan)
    
    d1[valid] = (np.log(F[valid] / K[valid]) + 0.5 * sigma[valid]**2 * T[valid]) / (sigma[valid] * np.sqrt(T[valid]))
    d2[valid] = d1[valid] - sigma[valid] * np.sqrt(T[valid])
    
    discount = np.exp(-r * T)
    
    # Compute call prices
    is_call = option_type == 'C'
    call_mask = valid & is_call
    if np.any(call_mask):
        prices[call_mask] = discount[call_mask] * (
            F[call_mask] * norm.cdf(d1[call_mask]) - K[call_mask] * norm.cdf(d2[call_mask])
        )
    
    # Compute put prices
    is_put = option_type == 'P'
    put_mask = valid & is_put
    if np.any(put_mask):
        prices[put_mask] = discount[put_mask] * (
            K[put_mask] * norm.cdf(-d2[put_mask]) - F[put_mask] * norm.cdf(-d1[put_mask])
        )
    
    return prices


def black76_implied_volatility_vectorized(
    market_price, F, K, T, r, option_type,
    max_iter=100, tol=1e-6, initial_guess=0.3, verbose=False
):
    """
    Vectorized Newton-Raphson method for Black-76 implied volatility.
    
    Parameters:
    - market_price: numpy array of observed market prices
    - F: numpy array of forward prices
    - K: numpy array of strike prices
    - T: numpy array of time to expiry (in years)
    - r: risk-free rate (scalar or array)
    - option_type: numpy array of 'C' or 'P'
    - max_iter: maximum iterations for Newton-Raphson
    - tol: convergence tolerance
    - initial_guess: starting volatility guess
    - verbose: if True, print detailed diagnostic information
    
    Returns: numpy array of implied volatilities (NaN for failures)
    """
    # Convert to numpy arrays
    market_price = np.asarray(market_price, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    option_type = np.asarray(option_type)
    
    # Initialize output with NaN
    iv = np.full_like(market_price, np.nan, dtype=float)
    
    # Create valid mask (positive values and T > 0)
    valid = (market_price > 0) & (F > 0) & (K > 0) & (T > 0)
    
    # Check intrinsic value
    is_call = option_type == 'C'
    intrinsic = np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    discount = np.exp(-r * T)
    
    # Market price must be at least intrinsic value (with small tolerance)
    valid &= (market_price >= intrinsic * discount * 0.99)
    
    if not np.any(valid):
        return iv
    
    # Initialize sigma guess for valid entries using Brenner-Subrahmanyam approximation
    # sigma ≈ sqrt(2*pi/T) * (C/F) for ATM options, adjusted for moneyness
    sigma = np.full_like(market_price, initial_guess, dtype=float)
    
    # Use smarter initial guess for valid entries
    if np.any(valid):
        # Brenner-Subrahmanyam approximation (works well for near-ATM)
        atm_approx = np.sqrt(2 * np.pi / T[valid]) * (market_price[valid] / F[valid])
        # Clip to reasonable range and use as initial guess
        sigma[valid] = np.clip(atm_approx, 0.05, 2.0)
    
    # Store initial sigma for convergence check later
    sigma_initial = sigma.copy()
    
    # Track which entries are still being updated
    active = valid.copy()
    
    # Diagnostic tracking
    if verbose:
        iterations_taken = np.zeros(len(market_price), dtype=int)
        low_vega_count = np.zeros(len(market_price), dtype=int)
        hit_lower_bound = np.zeros(len(market_price), dtype=bool)
        hit_upper_bound = np.zeros(len(market_price), dtype=bool)
    
    for iteration in range(max_iter):
        if not np.any(active):
            break
        
        # Compute model price and vega for active entries
        model_price = black76_price_vectorized(
            F[active], K[active], T[active], r, sigma[active], option_type[active]
        )
        
        vega = black76_vega(
            F[active], K[active], T[active], r, sigma[active]
        )
        
        # Compute price difference
        price_diff = model_price - market_price[active]
        
        # Check convergence
        converged = np.abs(price_diff) < tol
        
        # Track diagnostics
        if verbose:
            active_indices = np.where(active)[0]
            iterations_taken[active_indices] = iteration + 1
            low_vega_mask = vega <= 1e-10
            low_vega_count[active_indices[low_vega_mask]] += 1
        
        # Update sigma using Newton-Raphson where not converged and vega is non-zero
        not_converged = ~converged & (vega > 1e-10)
        
        if np.any(not_converged):
            # Create temporary array for updates
            sigma_update = sigma[active].copy()
            sigma_before_clip = sigma_update.copy()
            sigma_update[not_converged] -= price_diff[not_converged] / vega[not_converged]
            
            # Clip sigma to reasonable bounds (avoid too-low values that cause numerical issues)
            sigma_update = np.clip(sigma_update, 0.01, 5.0)
            
            # Track bound hits
            if verbose:
                active_indices = np.where(active)[0]
                hit_lower = (sigma_before_clip[not_converged] - price_diff[not_converged] / vega[not_converged]) < 0.01
                hit_upper = (sigma_before_clip[not_converged] - price_diff[not_converged] / vega[not_converged]) > 5.0
                hit_lower_bound[active_indices[np.where(not_converged)[0][hit_lower]]] = True
                hit_upper_bound[active_indices[np.where(not_converged)[0][hit_upper]]] = True
            
            # Update sigma for active entries
            sigma[active] = sigma_update
        
        # Mark converged entries as inactive
        active_indices = np.where(active)[0]
        active[active_indices[converged]] = False
    
    # Store successfully converged results
    converged_mask = valid & ~active
    iv[converged_mask] = sigma[converged_mask]
    
    # For entries that didn't converge, check if they're actually close to solution
    # AND have moved significantly from initial guess (avoid accepting 0.3 artifacts)
    still_active = valid & active
    accepted_near_convergence = np.zeros(len(market_price), dtype=bool)
    
    if np.any(still_active):
        final_price = black76_price_vectorized(
            F[still_active], K[still_active], T[still_active], r, 
            sigma[still_active], option_type[still_active]
        )
        relative_error = np.abs(final_price - market_price[still_active]) / market_price[still_active]
        
        # Much stricter criteria: must be within 1% error AND moved significantly from initial guess
        moved_from_initial = np.abs(sigma[still_active] - sigma_initial[still_active]) > 0.02
        close_enough = (relative_error < 0.01) & moved_from_initial
        
        if np.any(close_enough):
            still_active_indices = np.where(still_active)[0]
            iv[still_active_indices[close_enough]] = sigma[still_active][close_enough]
            accepted_near_convergence[still_active_indices[close_enough]] = True
    
    # Print detailed diagnostics
    if verbose:
        total_entries = len(market_price)
        rejected_intrinsic = (~valid).sum()
        converged_count = converged_mask.sum()
        near_converged_count = accepted_near_convergence.sum()
        failed_count = (valid & ~converged_mask & ~accepted_near_convergence).sum()
        
        print(f"\n    ═══ IV Solver Diagnostics ═══")
        print(f"    Total options: {total_entries:,}")
        print(f"    ├─ Rejected (below intrinsic): {rejected_intrinsic:,} ({rejected_intrinsic/total_entries*100:.1f}%)")
        print(f"    ├─ Converged successfully: {converged_count:,} ({converged_count/total_entries*100:.1f}%)")
        print(f"    ├─ Accepted near-convergence: {near_converged_count:,} ({near_converged_count/total_entries*100:.1f}%)")
        print(f"    └─ Failed to converge: {failed_count:,} ({failed_count/total_entries*100:.1f}%)")
        
        if failed_count > 0:
            failed_mask = valid & ~converged_mask & ~accepted_near_convergence
            
            # Analyze failure reasons
            print(f"\n    Failure Analysis:")
            
            # 1. Low vega (stuck)
            low_vega_failures = failed_mask & (low_vega_count > max_iter * 0.5)
            print(f"    ├─ Low vega (>50% iters): {low_vega_failures.sum():,} ({low_vega_failures.sum()/failed_count*100:.1f}%)")
            
            # 2. Hit bounds
            hit_lower_failures = failed_mask & hit_lower_bound
            hit_upper_failures = failed_mask & hit_upper_bound
            print(f"    ├─ Hit lower bound (0.01): {hit_lower_failures.sum():,} ({hit_lower_failures.sum()/failed_count*100:.1f}%)")
            print(f"    ├─ Hit upper bound (5.0): {hit_upper_failures.sum():,} ({hit_upper_failures.sum()/failed_count*100:.1f}%)")
            
            # 3. Max iterations
            max_iter_failures = failed_mask & (iterations_taken >= max_iter)
            print(f"    └─ Hit max iterations: {max_iter_failures.sum():,} ({max_iter_failures.sum()/failed_count*100:.1f}%)")
            
            # Compute final errors for failed entries
            failed_indices = np.where(failed_mask)[0]
            if len(failed_indices) > 0:
                final_prices_failed = black76_price_vectorized(
                    F[failed_mask], K[failed_mask], T[failed_mask], r,
                    sigma[failed_mask], option_type[failed_mask]
                )
                final_errors = np.abs(final_prices_failed - market_price[failed_mask]) / market_price[failed_mask]
                
                print(f"\n    Final relative errors (failed entries):")
                print(f"    ├─ Mean: {final_errors.mean():.2%}")
                print(f"    ├─ Median: {np.median(final_errors):.2%}")
                print(f"    └─ 90th percentile: {np.percentile(final_errors, 90):.2%}")
            
            # Sample failures
            print(f"\n    Sample failures (first 3):")
            for i, idx in enumerate(failed_indices[:3]):
                moneyness = np.log(K[idx] / F[idx])
                intrinsic = max(F[idx] - K[idx], 0) if option_type[idx] == 'C' else max(K[idx] - F[idx], 0)
                final_p = black76_price_vectorized(
                    np.array([F[idx]]), np.array([K[idx]]), np.array([T[idx]]), r,
                    np.array([sigma[idx]]), np.array([option_type[idx]])
                )[0]
                error = abs(final_p - market_price[idx]) / market_price[idx]
                
                print(f"    #{i+1}: {option_type[idx]}, K={K[idx]:.0f}, F={F[idx]:.0f}, T={T[idx]:.3f}y")
                print(f"        Moneyness: {moneyness:.3f}, Market: ${market_price[idx]:.2f}, Intrinsic: ${intrinsic:.2f}")
                print(f"        Final σ: {sigma[idx]:.4f}, Final price: ${final_p:.2f}, Error: {error:.2%}")
                print(f"        Iterations: {iterations_taken[idx]}, Low vega count: {low_vega_count[idx]}")
        
        # Convergence statistics for successful entries
        success_mask = converged_mask | accepted_near_convergence
        if success_mask.any():
            print(f"\n    Convergence Statistics (successful):")
            print(f"    ├─ Mean iterations: {iterations_taken[success_mask].mean():.1f}")
            print(f"    ├─ Median iterations: {np.median(iterations_taken[success_mask]):.0f}")
            print(f"    └─ Max iterations: {iterations_taken[success_mask].max()}")
    
    return iv