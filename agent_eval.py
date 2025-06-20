from data_generation import normalize_window

def evaluate_agent(agent, prices, initial_cash=1000):
    cash = initial_cash
    crypto = 0.0
    portfolio_history = []

    for i in range(20, len(prices)):
        price = prices[i]
        window = prices[i-20:i]
        inputs = normalize_window(window)

        agent.reset()
        agent.receive_inputs(inputs)
        agent.step(think=5)
        output = agent.decide_action()

        # Clamp output to [-1, 1]
        output = max(-1.0, min(1.0, output))

        # Calculate portfolio value BEFORE action
        portfolio_value = cash + crypto * price
        target_allocation = (output + 1.0) / 2.0  # [-1,1] → [0,1]
        target_crypto_value = target_allocation * portfolio_value
        current_crypto_value = crypto * price
        delta = target_crypto_value - current_crypto_value

        if delta > 0:
            # Buy
            max_affordable = cash
            amount_to_buy = min(delta, max_affordable) / price
            crypto += amount_to_buy
            cash -= amount_to_buy * price
        else:
            # Sell
            amount_to_sell = min(-delta / price, crypto)
            crypto -= amount_to_sell
            cash += amount_to_sell * price

        # Save new portfolio value
        portfolio_history.append(cash + crypto * price)

    return portfolio_history[-1], portfolio_history


def evaluate_agent_multi(agent, price_df, initial_cash=1000):
    """Evaluate an agent across multiple coins.

    Parameters
    ----------
    agent : Agent
        The trading agent.
    price_df : pandas.DataFrame
        DataFrame where each column is the price series of a crypto asset.
    initial_cash : float, optional
        Starting portfolio value.

    Returns
    -------
    float
        Final portfolio value when trading each asset independently and
        rebalancing the cash between them sequentially.
    list
        List of portfolio histories per asset.
    """
    results = []
    for column in price_df.columns:
        final_val, history = evaluate_agent(agent, price_df[column].values,
                                            initial_cash=initial_cash)
        results.append((final_val, history))

    final_values = [r[0] for r in results]
    return sum(final_values) / len(final_values), [r[1] for r in results]
