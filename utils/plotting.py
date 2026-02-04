import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_bonsai_data(data_collector, header, experiment, style):
    """
    Plots time-series data collected during the simulation.
    
    Parameters:
    - data_collector: list of dictionaries or values collected per step
    - header: label describing the plotted metric
    - experiment: experiment identifier
    - style: boolean indicating whether data refers to bonsai styles
    """
    data = pd.DataFrame(data_collector)

    if style:
        style_names = ["Chokkan", "Moyogi", "Shakan", "Kengai", "Han-Kengai", "No style defined"]
        data.columns = style_names

    data.plot(kind='line')

    plt.title(f"{header} Evolution Over Time")
    plt.xlabel("Step")
    plt.ylabel("Quantity")
    plt.legend(title=f"{header}")
    plt.grid(True)

    current_directory = os.getcwd()
    save_dir = os.path.join(current_directory, "Results")
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f"{header}_Experiment_{experiment}.png")

    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()

def plot_height_curvature_inclination(model, agent_data, experiment):
    """
    Plots the evolution of inclination, height, and curvature
    for all bonsai agents and computes global statistics.
    """
    bonsai_ids = [agent.unique_id for agent in model.schedule.agents if getattr(agent, "tipo", None) == "bonsai"]
    bonsai_data = agent_data[agent_data.index.get_level_values('AgentID').isin(bonsai_ids)]

    attributes = ["Inclination", "Height", "Curvature"]

    current_directory = os.getcwd()
    save_dir = os.path.join(current_directory, "Results")
    os.makedirs(save_dir, exist_ok=True)

    stats_dict = {}

    for attr in attributes:
        plt.figure(figsize=(10, 5))

        all_values = []

        for bonsai_id in bonsai_ids:
            bonsai_attr = bonsai_data.xs(bonsai_id, level='AgentID')[attr]
            plt.plot(bonsai_attr.index, bonsai_attr, label=f'Bonsai {bonsai_id}')
            all_values.append(bonsai_attr.values)

        plt.title(f"{attr} Evolution Over Time")
        plt.xlabel("Step")
        plt.ylabel(attr)
        plt.legend()
        plt.grid(True)

        file_name = os.path.join(save_dir, f"{attr}_Experiment_{experiment}.png")
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close()

        all_values = np.concatenate(all_values) 

        mean_value = np.mean(all_values)
        median_value = np.median(all_values)
        std_value = np.std(all_values)
        min_value = np.min(all_values)
        max_value = np.max(all_values)

        stats_dict[attr] = {
            "Mean": mean_value,
            "Median": median_value,
            "Std Dev": std_value,
            "Min": min_value,
            "Max": max_value
        }
    
    stats_file = os.path.join(save_dir, f"Inclination_Height_Curvature_total_stats_Experiment_{experiment}.csv")
    stats_df = pd.DataFrame(stats_dict).T
    stats_df.to_csv(stats_file)

def plot_reward_curve(caregivers):
    """
    Plots reward accumulation curves for each CaregiverAgent.
    Used to visualize learning convergence.
    """
    plt.figure(figsize=(10, 5))

    for agent, rewards in caregivers: 
        plt.plot(rewards, label=f"Caregiver {agent.unique_id}", alpha=0.7)

    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Acumulada")
    plt.title("Curva de Convergência dos CaregiverAgents")
    plt.legend()
    plt.show()
    
def plot_style_evolution(all_style_experiences):
    """
    Plots the evolution of caregiver experience for each bonsai style.
    """
    
    style_indices = {
        "Chokkan": 0,
        "Moyogi": 1,
        "Shakan": 2,
        "Kengai": 3,
        "Han-Kengai": 4,
        "No style defined": 5
    }
    
    all_style_experiences = np.array(all_style_experiences)
    
    transposed_arrays = all_style_experiences.T 

    plt.figure(figsize=(10, 5))

    for i, style_array in enumerate(transposed_arrays[:-1]):
        style_name = list(style_indices.keys())[i]
        
        plt.plot(style_array, label=f"{style_name}", marker='o')

    plt.xlabel("Iterations (Steps)")
    plt.ylabel("Experience Value")
    plt.title("Style Evolution Over Time")

    plt.legend()
    plt.grid(True)

    plt.show()
