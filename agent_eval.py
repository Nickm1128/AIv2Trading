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
