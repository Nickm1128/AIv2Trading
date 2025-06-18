import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def generate_synthetic_data(length=100):
    """
    Generate realistic nonstationary cryptocurrency-like price data with:
    - Regime changes (bull/bear markets)
    - Volatility clustering
    - Mean reversion within regimes
    - Realistic return distributions
    """
    prices = [100.0]
    
    # Initialize regime and volatility state
    regime = 'neutral'  # 'bull', 'bear', 'neutral'
    regime_timer = 0
    base_volatility = 0.02  # 2% daily volatility
    volatility = base_volatility
    
    # Regime parameters
    regimes = {
        'bull': {'drift': 0.001, 'vol_multiplier': 1.2, 'mean_duration': 30},
        'bear': {'drift': -0.0008, 'vol_multiplier': 1.5, 'mean_duration': 25},
        'neutral': {'drift': 0.0002, 'vol_multiplier': 1.0, 'mean_duration': 40}
    }
    
    for i in range(length - 1):
        current_price = prices[-1]
        
        # Regime switching logic
        regime_timer += 1
        regime_params = regimes[regime]
        
        # Probability of regime change increases over time
        change_prob = min(0.02 + regime_timer / (regime_params['mean_duration'] * 2), 0.15)
        
        if np.random.random() < change_prob:
            # Switch regime
            old_regime = regime
            regime = np.random.choice(['bull', 'bear', 'neutral'])
            regime_timer = 0
            regime_params = regimes[regime]
            
        # Volatility clustering (GARCH-like behavior)
        volatility_persistence = 0.85
        volatility_mean_reversion = 0.1
        volatility_shock = 0.05
        
        volatility = (volatility_persistence * volatility + 
                     volatility_mean_reversion * base_volatility * regime_params['vol_multiplier'] +
                     volatility_shock * abs(np.random.normal(0, 1)) * base_volatility)
        
        # Bound volatility to reasonable ranges
        volatility = np.clip(volatility, 0.005, 0.15)  # 0.5% to 15% daily vol
        
        # Generate return with regime-dependent drift
        base_return = np.random.normal(regime_params['drift'], volatility)
        
        # Add occasional jumps (fat tails)
        if np.random.random() < 0.02:  # 2% chance of jump
            jump_size = np.random.normal(0, 0.05) * (1 if np.random.random() > 0.5 else -1)
            base_return += jump_size
        
        # Add mean reversion within regime (prevents prices from going too extreme)
        if regime == 'bull' and current_price > 300:
            base_return -= 0.002  # Stronger pullback when overextended
        elif regime == 'bear' and current_price < 30:
            base_return += 0.003  # Support bounce when oversold
            
        # Apply geometric return (more realistic for prices)
        new_price = current_price * np.exp(base_return)
        
        # Ensure price stays positive with a reasonable floor
        new_price = max(1.0, new_price)
        
        prices.append(new_price)
    
    return np.array(prices)

def generate_multiple_scenarios(n_scenarios=5, length=200):
    """Generate multiple scenarios for robust backtesting"""
    scenarios = []
    for _ in range(n_scenarios):
        scenario = generate_synthetic_data(length)
        scenarios.append(scenario)
    return scenarios

def add_market_microstructure_noise(prices, noise_level=0.001):
    """Add realistic bid-ask bounce and microstructure noise"""
    noisy_prices = prices.copy()
    for i in range(1, len(prices)):
        # Bid-ask bounce
        if np.random.random() > 0.5:
            noisy_prices[i] *= (1 + np.random.uniform(0, noise_level))
        else:
            noisy_prices[i] *= (1 - np.random.uniform(0, noise_level))
    return noisy_prices

def normalize_window(window):
    diffs = np.diff(window)
    return diffs / (np.std(diffs) + 1e-6)
