import os
import pandas as pd

communication_labels = {
        0: "No_Communication",
        1: "Basic_Communication",
        2: "Communication_Evaluation"
    }

knowledgment_labels = {
        0: "No_Knowledge",
        1: "Some_Knowledge",
        2: "Expert"
    }

def calculate_statistics(data_collector, name):
    """
    Computes basic descriptive statistics from a time-series.
    """
    if not data_collector: 
        print(f"Nenhum dado disponível para {name}.")
        return None

    data_series = pd.Series(data_collector) 

    statistics = {
        "Média": data_series.mean(),
        "Mediana": data_series.median(),
        "Desvio Padrão": data_series.std(),
        "Mínimo": data_series.min(),
        "Máximo": data_series.max()
    }

    return statistics

def calculate_style_statistics(data_collector_styles):
    """
    Computes descriptive statistics for bonsai styles over time.
    """
    df = pd.DataFrame(data_collector_styles)

    statistics = df.describe() 

    return statistics

def save_death_health_style(stats_deaths, stats_health, stats_styles, experiment):
    """
    Saves summary statistics of deaths, health, and styles to text files.
    """
    current_directory = os.getcwd()
    save_dir = os.path.join(current_directory, "Results")
    os.makedirs(save_dir, exist_ok=True)

    stats_deaths_file = os.path.join(save_dir, f"deaths_Experiment_{experiment}.txt")
    stats_health_file = os.path.join(save_dir, f"health_Experiment_{experiment}.txt")
    stats_styles_file = os.path.join(save_dir, f"style_Experiment_{experiment}.txt")

    with open(stats_deaths_file, 'w') as file:
        file.write("Deaths Statistics:\n")
        for key, value in stats_deaths.items():
            file.write(f"{key}: {value}\n")

    with open(stats_health_file, 'w') as file:
        file.write("Health Statistics:\n")
        for key, value in stats_health.items():
            file.write(f"{key}: {value}\n")

    with open(stats_styles_file, 'w') as file:
        file.write("Style Statistics:\n")
        for style, stats in stats_styles.items():
            file.write(f"\n{style}:\n")
            for stat_key, stat_value in stats.items():
                file.write(f"  {stat_key}: {stat_value}\n")

