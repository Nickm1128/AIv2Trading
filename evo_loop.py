from Agent import Agent
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import multiprocessing

PROJECT_ROOT = Path(__file__).resolve().parent

from data_generation import (
    load_real_crypto_data,
    sample_price_windows,
)
from agent_eval import evaluate_agent
from mutations import mutate_agent


def _evaluate_agent_worker(args):
    """Helper function for multiprocessing Pool."""
    agent, scenarios = args
    scenario_scores = []
    for scenario in scenarios:
        try:
            coin_scores = []
            for coin in scenario.columns:
                prices = scenario[coin].values
                final_value, _ = evaluate_agent(agent, prices)
                buy_hold_return = prices[-1] / prices[0]
                agent_return = final_value / 1000.0
                coin_scores.append(agent_return / buy_hold_return)
            scenario_scores.append(np.mean(coin_scores))
        except Exception:
            scenario_scores.append(0.5)

    median_score = np.median(scenario_scores)
    return agent, median_score, scenario_scores

def create_population(n_agents, n_neurons):
    return [Agent(f"agent_{i}", n_neurons) for i in range(n_agents)]


def evolve(pop_size=5, generations=10_000, base_neurons=10, mutation_rate=0.1,
           mutation_strength=0.1, processes=None, population_path=None):
    """Run the evolutionary loop.

    Parameters
    ----------
    pop_size : int
        Number of agents in the population.
    generations : int
        Number of generations to evolve.
    base_neurons : int
        Starting neuron count for each agent.
    mutation_rate : float
        Probability of mutating a synapse or neuron parameter.
    mutation_strength : float
        Magnitude of mutation applied.
    processes : int or None
        Number of processes to use for evaluation. ``None`` defaults to using
        all available CPUs.
    """

    if population_path:
        with open(population_path, "rb") as f:
            population = pickle.load(f)
            population = population[:pop_size]
    else:
        population = [Agent(f"agent_{i}", base_neurons) for i in range(pop_size)]

    if processes is None:
        processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=processes) if processes > 1 else None

    # Load real crypto data once at the start and split into train/test
    train_df, test_df = load_real_crypto_data(split=True)
    all_windows = sample_price_windows(train_df, n_samples=3, window_size=300)

    # Track performance over generations
    best_scores_history = []
    avg_scores_history = []
    
    for gen in range(generations):
        if gen % 20 == 0:
            best_scores_history = []
            avg_scores_history = []
            all_windows = sample_price_windows(train_df, n_samples=3, window_size=300)
            
        scenarios = all_windows

        # Evaluate each agent across all scenarios
        if pool:
            eval_results = pool.map(
                _evaluate_agent_worker,
                [(agent, scenarios) for agent in population]
            )
        else:
            eval_results = [
                _evaluate_agent_worker((agent, scenarios)) for agent in population
            ]

        agent_scores = [(agent, score) for agent, score, _ in eval_results]
        score_lookup = {agent: scores for agent, score, scores in eval_results}
        
        # Sort by performance (higher is better)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract scores for reporting
        scores = [score for _, score in agent_scores]
        best_score = scores[0]
        avg_score = np.mean(scores)
        
        # Track history
        best_scores_history.append(best_score)
        avg_scores_history.append(avg_score)
        
        print(f"Gen {gen:2d} - Best: {best_score:.3f} - Avg: {avg_score:.3f} - "
              f"Std: {np.std(scores):.3f}")
        
        # Selection: keep top performers
        n_survivors = min(5, pop_size)
        top_entries = agent_scores[:n_survivors]

        survivors = []
        for agent, score in top_entries:
            parent_scores = score_lookup[agent]
            child = mutate_agent(agent, 0.01, mutation_strength)
            _, child_score, child_scores = _evaluate_agent_worker((child, scenarios))
            try:
                from scipy.stats import ttest_rel
                _, p_val = ttest_rel(child_scores, parent_scores)
            except Exception:
                p_val = 1.0
            if child_score > score and p_val < 0.05:
                survivors.append(child)
            else:
                survivors.append(agent)

        # Create additional children to restore population size
        children = []
        while len(children) < pop_size - n_survivors:
            parent = random.choice(survivors)
            child = mutate_agent(parent, 0.01, mutation_strength)
            children.append(child)

        population = survivors + children
    
    # --- Final evaluation on fresh data ---
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    # Generate completely fresh scenarios for final evaluation from the
    # reserved test data
    final_scenarios = sample_price_windows(test_df, n_samples=5, window_size=300)
    
    # Evaluate top 3 agents
    top_agents = [agent for agent, _ in agent_scores[:3]]
    
    final_results = []
    for i, agent in enumerate(top_agents):
        scenario_results = []
        for j, scenario in enumerate(final_scenarios):
            coin_returns = []
            for coin in scenario.columns:
                prices = scenario[coin].values
                final_value, portfolio_history = evaluate_agent(agent, prices)
                buy_hold_value = 1000 * (prices[-1] / prices[0])
                agent_return = (final_value / 1000.0 - 1) * 100
                buy_hold_return = (buy_hold_value / 1000.0 - 1) * 100

                scenario_results.append({
                    'final_value': final_value,
                    'portfolio_history': portfolio_history,
                    'agent_return': agent_return,
                    'buy_hold_return': buy_hold_return,
                    'scenario': scenario[coin].values,
                    'coin': coin
                })
                coin_returns.append(agent_return)
        
        # Calculate statistics
        agent_returns = [r['agent_return'] for r in scenario_results]
        buy_hold_returns = [r['buy_hold_return'] for r in scenario_results]
        
        print(f"\nAgent {i+1} ({agent.name}):")
        print(f"  Avg Return: {np.mean(agent_returns):6.2f}% (vs {np.mean(buy_hold_returns):6.2f}% buy-hold)")
        print(f"  Std Return: {np.std(agent_returns):6.2f}% (vs {np.std(buy_hold_returns):6.2f}% buy-hold)")
        print(f"  Win Rate:   {np.mean([r > 0 for r in agent_returns]):6.2f} ({np.sum([ar > bh for ar, bh in zip(agent_returns, buy_hold_returns)])}/5 vs buy-hold)")
        
        final_results.append({
            'agent': agent,
            'scenario_results': scenario_results,
            'avg_return': np.mean(agent_returns)
        })
    
    # --- Save best agent and results ---
    best_agent = final_results[0]['agent']
    with open(PROJECT_ROOT / "final_population.pkl", "wb") as f:
        pickle.dump(population, f)

    with open(PROJECT_ROOT / "best_agent.pkl", "wb") as f:
        pickle.dump(best_agent, f)

    with open(PROJECT_ROOT / "evolution_results.pkl", "wb") as f:
        pickle.dump({
            'final_results': final_results,
            'best_scores_history': best_scores_history,
            'avg_scores_history': avg_scores_history
        }, f)
    
    # --- Plot results ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Evolution progress
    axes[0, 0].plot(best_scores_history, label='Best', linewidth=2)
    axes[0, 0].plot(avg_scores_history, label='Average', linewidth=2)
    axes[0, 0].set_title('Evolution Progress')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Relative Performance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Best agent on first scenario
    scenario_0 = final_results[0]['scenario_results'][0]
    axes[0, 1].plot(scenario_0['scenario'][3:], label='Price', alpha=0.7)
    axes[0, 1].plot(scenario_0['portfolio_history'], label='Agent Portfolio', linewidth=2)
    buy_hold_portfolio = [1000 * (price / scenario_0['scenario'][3]) for price in scenario_0['scenario'][3:]]
    axes[0, 1].plot(buy_hold_portfolio, label='Buy & Hold', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Best Agent Performance (Scenario 1)')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Return distribution
    all_agent_returns = [r['agent_return'] for result in final_results for r in result['scenario_results']]
    all_buy_hold_returns = [r['buy_hold_return'] for result in final_results for r in result['scenario_results']]
    
    axes[1, 0].hist(all_agent_returns, bins=10, alpha=0.7, label='Agent Returns', density=True)
    axes[1, 0].hist(all_buy_hold_returns, bins=10, alpha=0.7, label='Buy-Hold Returns', density=True)
    axes[1, 0].set_title('Return Distribution')
    axes[1, 0].set_xlabel('Return %')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Agent vs Buy-Hold scatter
    agent_rets = [r['agent_return'] for r in final_results[0]['scenario_results']]
    bh_rets = [r['buy_hold_return'] for r in final_results[0]['scenario_results']]
    
    axes[1, 1].scatter(bh_rets, agent_rets, s=100, alpha=0.7)
    axes[1, 1].plot([-20, 20], [-20, 20], 'r--', alpha=0.5, label='Equal Performance')
    axes[1, 1].set_title('Agent vs Buy-Hold Returns')
    axes[1, 1].set_xlabel('Buy-Hold Return %')
    axes[1, 1].set_ylabel('Agent Return %')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "evolution_summary.png", dpi=300)
    plt.show()

    if pool:
        pool.close()
        pool.join()

    return best_agent, final_results

if __name__ == '__main__':
    evolve(population_path=PROJECT_ROOT / "final_population.pkl")
