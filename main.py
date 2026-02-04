import os
import random
import pickle
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from mesa import Agent, Model
from mesa.time import RandomActivation, StagedActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from config.parameters import *
from agents.bonsai_agent import BonsaiAgent
from agents.caregiver_agent import CaregiverAgent
from model.bonsai_model import BonsaiModel

from utils.plotting import plot_bonsai_data, plot_height_curvature_inclination, plot_style_evolution
from utils.statistics import calculate_statistics, calculate_style_statistics, save_death_health_style

def run_experiment(experiment_number, num_bonsai=30, num_caregivers=1, width=2, height=2, knowledgement=1, wiring_strategy="full"):
    """Runs a simulation experiment and returns the collected data."""
    model = BonsaiModel(num_bonsai=num_bonsai, num_caregivers=num_caregivers,
                        width=width, height=height, knowledgement=knowledgement,
                        wiring_strategy=wiring_strategy)
    
    for step in range(YEAR * YEAR_QUANTITY):
        model.step()

    stats_deaths = calculate_statistics(model.data_collector_deaths, "Mortes dos Bonsais")
    stats_health = calculate_statistics(model.data_collector_health, "Saúde dos Bonsais")
    stats_styles = calculate_style_statistics(model.data_collector_styles)

    save_death_health_style(stats_deaths, stats_health, stats_styles, experiment_number)

    plot_bonsai_data(model.data_collector_styles, "Styles", experiment_number, False)
    plot_bonsai_data(model.data_collector_deaths, "Deaths", experiment_number, False)
    plot_bonsai_data(model.data_collector_health, "Health", experiment_number, False)
    plot_height_curvature_inclination(model, model.datacollector.get_agent_vars_dataframe(), experiment_number)

    return {
        "deaths": model.data_collector_deaths,
        "health": model.data_collector_health,
        "styles": model.data_collector_styles
    }


def main():
    random.seed(2025)
    NUM_EXPERIMENTS = 10

    all_data = {"deaths": [], "health": [], "styles": []}

    all_data_mean = {"deaths": [], "health": [], "styles": []}

    for exp_num in range(NUM_EXPERIMENTS):
        print(f"🌳 Running bonsai model {exp_num+1}/{NUM_EXPERIMENTS} 🌳")
        data = run_experiment(exp_num)
        
        for key in all_data:
            all_data[key].extend(data[key])
            all_data_mean[key].append(data[key])


    stats_deaths_calculated = calculate_statistics(all_data["deaths"], "Mortes dos Bonsais")
    stats_health_calculated = calculate_statistics(all_data["health"], "Saúde dos Bonsais")
    stats_styles_calculated = calculate_style_statistics(all_data["styles"])

    save_death_health_style(stats_deaths_calculated, stats_health_calculated, stats_styles_calculated, "Final_Stats")

    style_order = ["Chokkan", "Moyogi", "Shakan", "Kengai", "Han-Kengai", "No style defined"]
    num_steps = len(all_data_mean["styles"][0])
    averaged_styles_per_step = []

    for step_idx in range(num_steps):
        step_totals = {style: 0 for style in style_order}
        num_instances = len(all_data_mean["styles"])

        for instance in all_data_mean["styles"]:
            for style in style_order:
                step_totals[style] += instance[step_idx][style]

        step_avg = [step_totals[style] / num_instances for style in style_order]
        averaged_styles_per_step.append(step_avg)

    averaged_styles_per_step = np.array(averaged_styles_per_step)
    average_deaths = np.mean(np.array(all_data_mean["deaths"]), axis=0)
    average_health = np.mean(np.array(all_data_mean["health"]), axis=0)

    plot_bonsai_data(averaged_styles_per_step, "Styles", "Final_Stats", True)
    plot_bonsai_data(average_deaths, "Deaths", "Final_Stats", False)
    plot_bonsai_data(average_health, "Health", "Final_Stats", False)


if __name__ == "__main__":
    main()