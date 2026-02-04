from mesa import Model
from mesa.time import RandomActivation, StagedActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from agents.bonsai_agent import BonsaiAgent
from agents.caregiver_agent import CaregiverAgent

from config.parameters import *
import random
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

class BonsaiModel(Model):
    def __init__(self, num_bonsai=0, num_caregivers=0, width=0, height=0, knowledgement = 0, wiring_strategy="full" ):

        # ============================================================
        # ===== Experimental parameters =====
        # ============================================================
        self.knowledgement = knowledgement
        self.wiring_strategy = wiring_strategy

        # ============================================================
        # ===== Environment =====
        # ===========================================================
        self.num_bonsai = num_bonsai
        self.num_caregivers = num_caregivers
        self.width = width
        self.height = height

        self.wind_strength = random.uniform(0, 4)
        self.light_strength = random.uniform(0, 4)

        self.rain = False
        self.rain_season = 0

        self.season = "spring"
        self.water_consumption = 1

        # ============================================================
        # ===== Mesa =====
        # ============================================================
        self.step_count = 0
        self.schedule = RandomActivation(self)
        self.space = MultiGrid(width, height, torus=True)
        self.data_collector_styles = []
        self.data_collector_deaths = []
        self.data_collector_health = []
        self.schedule = StagedActivation(self, stage_list=["caregiver_step", "bonsai_step"])

        self.datacollector = DataCollector(
            model_reporters={
           
            "BonsaiStyles": self.get_bonsai_styles
            },
            agent_reporters={
            "Style": lambda a: a.style if hasattr(a, "style") else None,
            "Inclination": lambda a: a.inclination if hasattr(a, "inclination") else None,
            "Height": lambda a: a.height if hasattr(a, "height") else None,
            "Branches": lambda a: a.branches if hasattr(a, "branches") else None,
            "Curvature": lambda a: a.curvature if hasattr(a, "curvature") else None,
            }

        )


        # ==========================================================
        # ================== Create Bonsai Agents ==================
        # ==========================================================
        bonsais = []

        for i in range(num_bonsai):
            bonsai = BonsaiAgent(i, self)
            self.schedule.add(bonsai)
            self.space.place_agent(bonsai, (0, 0))
            bonsais.append(bonsai)


        # ============================================================
        # ================== Create Caregiver Agents =================
        # ============================================================
        for i in range(num_caregivers):
            caregiver = CaregiverAgent(
            num_bonsai + i,
            self,
            influence=1,
            bonsais=bonsais,
            actions_per_day=ACTIONS_PER_DAY
            )
            self.schedule.add(caregiver)
            self.space.place_agent(caregiver, (0, 1))

    def get_caregivers(self):
        """
        Returns a dictionary mapping each CaregiverAgent to its reward history.
        Useful for post-simulation analysis of learning dynamics.
        """
        caregivers = {}

        for agent in self.schedule.agents:
            if isinstance(agent, CaregiverAgent):
                caregivers[agent] = agent.reward_per_step

        return caregivers
    
    def get_caregivers_style_experiences(self):
        """
        Collects the evolution of style-specific experience for each caregiver.
        Each entry corresponds to one caregiver and stores experience per year.
        """
        caregivers_style_experience = []

        for agent in self.schedule.agents:
            if isinstance(agent, CaregiverAgent):
                caregivers_style_experience.append(agent.experience_per_year_by_style)

        return caregivers_style_experience

    def get_bonsai_styles(self):
        """
        Counts how many bonsais currently belong to each classical style.
        Used as a global descriptive statistic of the population.
        """
        style_count = {
            "Chokkan": 0,
            "Moyogi": 0,
            "Shakan": 0,
            "Kengai": 0,
            "Han-Kengai": 0
        }
        style_count = {
            "Chokkan": 0,
            "Moyogi": 0,
            "Shakan": 0,
            "Kengai": 0,
            "Han-Kengai": 0
        }

        for agent in self.schedule.agents:
            if getattr(agent, "tipo", None) == "bonsai":
                if agent.style in style_count:
                    style_count[agent.style] += 1

        return style_count

    def get_bonsai_deaths(self):
        """
        Counts the total number of dead bonsai agents.
        This metric is used to evaluate system sustainability.
        """
        death_count = 0

        for agent in self.schedule.agents:
            if getattr(agent, "tipo", None) == "bonsai":
                if agent.status == 'dead':
                    death_count += 1

        return death_count

    def get_bonsai_health(self):
        """
        Computes the average health of all bonsai agents currently alive.
        """
        health_count = 0

        for agent in self.schedule.agents:
            if getattr(agent, "tipo", None) == "bonsai":
                    health_count += agent.health

        return health_count / self.num_bonsai

    def update_season(self):
        """
        Updates the current season based on simulation time.
        Each season modifies water consumption and growth rates.
        """
        seasons = [
            {"season": "spring", "water_consumption": 1, "growth_season": 0.2},
            {"season": "summer", "water_consumption": 1.5, "growth_season": 0.1},
            {"season": "autumn", "water_consumption": 0.8, "growth_season": 0.08},
            {"season": "winter", "water_consumption": 0.5, "growth_season": 0}
        ]

        season_index = int(self.step_count // (YEAR // 4)) % len(seasons)
        current_season = seasons[season_index]
        if current_season["season"] != self.season:
            self.rain_season = 0
        self.season = current_season["season"]
        self.water_consumption = current_season["water_consumption"]
        self.growth_season = current_season["growth_season"]

    def compute_average_style(self):
        """
        (Currently unused / placeholder)
        Intended to compute an aggregate style metric.
        WARNING: styles are strings, so this function is not meaningful as-is.
        """
        styles = [agent.style for agent in self.schedule.agents if getattr(agent, "tipo", None) == "bonsai"]
        return sum(styles) / len(styles) if styles else 0
    
    def get_bonsai_styles(self):
        """
        Returns a dictionary mapping each bonsai ID to its current style.
        Useful for tracking individual trajectories.
        """
        bonsai_styles =  {
            agent.unique_id: agent.style
            for agent in self.schedule.agents
                if getattr(agent, "tipo", None) == "bonsai"
        }   
        return dict(sorted(bonsai_styles.items()))

    def day_parameters(self):
        """
        Updates daily environmental parameters such as wind, light, and rainfall.
        These factors influence bonsai morphology and water availability.
        """
        SEASONAL_RAINFALL = {
        "spring": 350,
        "summer": 650,
        "autumn": 650,
        "winter": 81
        }

        WIND_PATTERNS = {
        "Jan": {"direction": "NW1", "pattern": ["Strong C curl", "local AC curl"]},
        "Feb": {"direction": "NW1", "pattern": ["Strong C curl", "local AC curl"]},
        "March": {"direction": "W1", "pattern": ["Strong C curl"]},
        "April": {"direction": "S-SE", "pattern": ["C curl", "local AC curl"]},
        "May": {"direction": "S-SE", "pattern": ["C curl", "local AC curl"]},
        "June": {"direction": "S-SE", "pattern": ["C curl", "local AC curl"]},
        "July": {"direction": "S-SE", "pattern": ["C curl", "local AC curl"]},
        "Aug": {"direction": "S-SE", "pattern": ["C curl", "local AC curl"]},
        "Sept": {"direction": "NE", "pattern": ["C curl", "AC curl"]},
        "Oct": {"direction": "NW2", "pattern": ["Mostly C/AC curl"]},
        "Nov": {"direction": "NW1", "pattern": ["Strong C/AC curl"]},
        "Dec": {"direction": "N", "pattern": ["Strong C/AC curl"]},
        }

        WIND_STRENGTH_RANGES = {
        "NW1": (0.0, 4.0),  
        "NW2": (0.0, 2.0),
        "N": (0.0, 4.0),
        "NE": (0.0, 4.0),  
        "W1": (0.0, 4.0),
        "S-SE": (0.0, 2.0),
        }

        current_month = list(WIND_PATTERNS.keys())[(self.step_count // 30) % 12]

        wind_data = WIND_PATTERNS[current_month]

        self.wind_direction = wind_data["direction"]
        self.wind_patterns = wind_data["pattern"]

        min_strength, max_strength = WIND_STRENGTH_RANGES.get(self.wind_direction, (0.0, 4.0))
        self.wind_strength = random.uniform(min_strength, max_strength)

        self.light_strength = random.uniform(0, 4)

        if random.random() < (4 / 91):
            base_rain = SEASONAL_RAINFALL[self.season]
            self.rain = True
            self.rain_season += random.uniform(base_rain * 0.2, base_rain * 0.3)
        else:
            self.rain = False
        
        if self.rain:
            rain_amount = random.uniform(base_rain * 0.2, base_rain * 0.3)  
            if rain_amount < 75:
                rain_type = "weak"
            elif rain_amount < 150:
                rain_type = "moderate"
            else:
                rain_type = "strong"

        for agent in self.schedule.agents:
            if getattr(agent, "tipo", None) == "bonsai":
                if self.rain:
                    agent.daily_care = True
                    if rain_type == "weak":
                        agent.increase_water(INCREASE_WATER_WATERING * 0.5)
                    elif rain_type == "moderate":
                        agent.increase_water(INCREASE_WATER_WATERING * 0.8)
                    else:
                        agent.increase_water(INCREASE_WATER_WATERING)
                else:
                    agent.daily_care = False

    def get_bonsai_styles(self):
        """
        Counts bonsais by style, including those without a defined style.
        NOTE: This method overrides previous definitions with the same name.
        """
        style_count = {
            "Chokkan": 0,
            "Moyogi": 0,
            "Shakan": 0,
            "Kengai": 0,
            "Han-Kengai": 0,
            "No style defined": 0
        }

        for agent in self.schedule.agents:
            if getattr(agent, "tipo", None) == "bonsai":
                if agent.style in style_count:
                    style_count[agent.style] += 1

        return style_count

    def q_table_agent(self):
        """
        Debugging utility to print Q-tables of all caregiver agents.
        """
        for agent in self.schedule.agents:
            if isinstance(agent, CaregiverAgent):
                print(f"Q-table do cuidador {agent.unique_id}: {agent.q_table}")
                
                for state, actions in agent.q_table.items():
                    print(f"Estado: {state}")
                    for action, q_value in actions.items():
                        print(f"Ação: {action}, Q-valor: {q_value}")

    def get_q_table(self):
        """
        Returns the Q-table and exploration rate (epsilon)
        of the first caregiver found.
        """
        for agent in self.schedule.agents:
            if isinstance(agent, CaregiverAgent):
                return agent.q_table, agent.epsilon

    def step(self):
        """
        Executes one simulation step:
        - Collects global statistics
        - Updates seasonal and environmental parameters
        - Advances all agents according to the schedule
        """
        bonsai_style = self.get_bonsai_styles()
        bonsai_death = self.get_bonsai_deaths()
        bonsai_health = self.get_bonsai_health()
        self.data_collector_styles.append(bonsai_style)
        self.data_collector_deaths.append(bonsai_death)
        self.data_collector_health.append(bonsai_health)

        self.step_count += 1
        self.update_season()
        self.day_parameters()
        self.datacollector.collect(self)
        self.schedule.step()
