from Agent import Agent, Neuron, Synapse
import numpy as np
import random
import copy

def mutate_agent(original_agent, mutation_rate, mutation_strength):
    """
    Creates a mutated copy of an agent, allowing for structural changes (adding neurons),
    modifying its parameters, and potentially pruning.
    """
    mutated_agent = copy.deepcopy(original_agent)

    # --- Structural Mutation: Add Neurons ---
    ADD_NEURON_CHANCE = 0.05 
    MAX_NEURONS_TO_ADD = 2 

    if random.random() < ADD_NEURON_CHANCE:
        num_neurons_to_add = random.randint(1, MAX_NEURONS_TO_ADD)
        current_neuron_count = len(mutated_agent.neurons)
        for i in range(num_neurons_to_add):
            new_neuron_name = f"{mutated_agent.name}_n_new{current_neuron_count + i}"
            new_neuron = Neuron(new_neuron_name)
            mutated_agent.neurons.append(new_neuron)
            
            num_new_synapses_for_neuron = random.randint(7, 15) 
            for _ in range(num_new_synapses_for_neuron):
                if len(mutated_agent.neurons) >= 2: 
                    pre, post = random.sample(mutated_agent.neurons, 2)
                    weight = random.uniform(-1, 1)
                    mutated_agent.synapses.append(Synapse(pre, post, weight))
    
    PRUNING_CHANCE = 0.1 
    PRUNING_NEURON_MAX = 1
    PRUNING_SYNAPSE_MAX_PERCENT = 0.05

    MIN_NEURONS = 4 

    if random.random() < PRUNING_CHANCE:
        neurons_to_prune_count = 1#random.randint(0, PRUNING_NEURON_MAX)
        actual_neurons_to_prune = min(neurons_to_prune_count, len(mutated_agent.neurons) - MIN_NEURONS)
        
        if actual_neurons_to_prune > 0:
            non_output_neurons = [n for n in mutated_agent.neurons if n not in mutated_agent.output_neurons]
            
            if len(non_output_neurons) > 0:
                neurons_to_remove = random.sample(non_output_neurons, min(actual_neurons_to_prune, len(non_output_neurons)))
                
                for neuron_to_remove in neurons_to_remove:
                    mutated_agent.neurons.remove(neuron_to_remove)
                    mutated_agent.synapses = [
                        syn for syn in mutated_agent.synapses 
                        if syn.pre != neuron_to_remove and syn.post != neuron_to_remove
                    ]


    if random.random() < PRUNING_CHANCE:
        num_synapses_to_prune = int(len(mutated_agent.synapses) * PRUNING_SYNAPSE_MAX_PERCENT)
        
        if num_synapses_to_prune > 0:
            synapses_to_remove = random.sample(mutated_agent.synapses, min(num_synapses_to_prune, len(mutated_agent.synapses)))
            for syn in synapses_to_remove:
                mutated_agent.synapses.remove(syn)

    for syn in mutated_agent.synapses:
        if random.random() < mutation_rate:
            syn.weight += random.uniform(-mutation_strength, mutation_strength)
            syn.weight = max(min(syn.weight, 1.0), -1.0)

    for neuron in mutated_agent.neurons:
        if random.random() < mutation_rate:
            neuron.threshold_pos += random.uniform(-0.1, 0.1)
            neuron.threshold_neg += random.uniform(-0.1, 0.1)
            neuron.threshold_pos = max(0.01, min(1.0, neuron.threshold_pos))
            neuron.threshold_neg = min(-0.01, max(-1.0, neuron.threshold_neg))

    mutated_agent.name = 'Child' + f"_mut{random.randint(0, 9999)}_NEURONCOUNT{len(mutated_agent.neurons)}"

    return mutated_agent
