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


def load_real_crypto_data(split=False, test_fraction=0.25):
    """Load time-synchronised crypto prices from the remote database.

    Parameters
    ----------
    split : bool, optional
        If ``True`` the returned value is a tuple ``(train_df, test_df)`` where
        the last ``test_fraction`` of the data is reserved for testing.  If
        ``False`` only the merged DataFrame is returned.
    test_fraction : float, optional
        Fraction of the most recent data to reserve for testing when
        ``split`` is ``True``.

    Returns
    -------
    pandas.DataFrame or Tuple[pandas.DataFrame, pandas.DataFrame]
        When ``split`` is ``False`` a single DataFrame indexed by timestamp
        containing closing prices for BTC, ETH, ADA, XRP and SOL is returned.
        If ``split`` is ``True`` the DataFrame is split chronologically and a
        tuple ``(train_df, test_df)`` is returned.
    """
    import pandas as pd
    from sqlalchemy import create_engine

    db_url = (
        "postgresql://technical_ping_db_user:"
        "y0YrjDroozyBP5kFykBPYOTE4EjDdhQK@"
        "dpg-d0j718buibrs73co3eag-a.singapore-postgres.render.com/technical_ping_db"
    )
    engine = create_engine(db_url)

    tables = [
        "crypto_prices_BTC_usd",
        "crypto_prices_ETH_usd",
        "crypto_prices_ADA_usd",
        "crypto_prices_XRP_usd",
        "crypto_prices_SOL_usd",
    ]
    table_prefix = "crypto_prices_"

    close_dfs = []

    for table in tables:
        with engine.connect() as conn:
            query = f"SELECT timestamp, close FROM {table}"
            df = pd.read_sql(query, con=conn)

        if "timestamp" not in df.columns or "close" not in df.columns:
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").dropna()
        df = df.drop_duplicates(subset="timestamp")
        df = df.set_index("timestamp")

        name = table[len(table_prefix) :].lower().replace("_usd", "")
        df = df.rename(columns={"close": name})

        close_dfs.append(df)

    merged = pd.concat(close_dfs, axis=1, join="inner")
    merged = merged.dropna().sort_index()

    if not split:
        return merged

    split_idx = int(len(merged) * (1 - test_fraction))
    train_df = merged.iloc[:split_idx]
    test_df = merged.iloc[split_idx:]
    return train_df, test_df


def sample_price_windows(price_df, n_samples=150, window_size=300):
    """Return a list of price windows from a merged price DataFrame."""
    if len(price_df) < window_size:
        raise ValueError("Not enough data for the requested window size")

    samples = []
    max_start = len(price_df) - window_size
    for _ in range(n_samples):
        start = np.random.randint(0, max_start + 1)
        samples.append(price_df.iloc[start : start + window_size])

    return samples
