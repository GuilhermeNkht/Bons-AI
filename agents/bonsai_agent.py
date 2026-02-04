from mesa import Agent
from config.parameters import *
import random
import numpy as np


class BonsaiAgent(Agent):
    """
    Represents a bonsai tree as an autonomous biological agent.

    This agent models physiological health, structural growth,
    and stylistic. Its state evolves over time under
    environmental dynamics and caregiver interventions.
    """

    PLANT_SPECIES = [
    {"name": "Ulmus paviflora", "prune_season": "spring", "water_need": 30, "water_max": 100, "repotting_year" : 1, "wire_min_year": 0.5, "wire_max_year": 1, "max_inclination": 360, "max_curvature": 5, "min_curvature": 0, "growth_rate": 0.7},
    {"name": "Ficus microcarpa", "prune_season": "spring", "water_need": 30, "water_max": 100, "repotting_year" : 1, "wire_min_year": 0.5, "wire_max_year": 1, "max_inclination": 360, "max_curvature": 5, "min_curvature": 0, "growth_rate": 0.8},
    {"name": "Buxus harlandii", "prune_season": "spring", "water_need": 30, "water_max": 100, "repotting_year" : 1, "wire_min_year": 0.5, "wire_max_year": 1, "max_inclination": 360, "max_curvature": 5, "min_curvature": 0, "growth_rate": 0.5},
    {"name": "Ficus virens", "prune_season": "spring", "water_need": 30, "water_max": 100, "repotting_year" : 1, "wire_min_year": 0.5, "wire_max_year": 1, "max_inclination": 360, "max_curvature": 5, "min_curvature": 0, "growth_rate": 0.75}
    ]

    SEASON = [
    {"season": "spring", "water_consumption": 1, "growth_season": 0.2},
    {"season": "summer", "water_consumption": 2, "growth_season": 0.1},
    {"season": "autumn", "water_consumption": 0.8, "growth_season": 0.08},
    {"season": "winter", "water_consumption": 0.5, "growth_season": 0}
    ]
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # ============================================================
        # Agent Identity
        # ============================================================
        self.tipo = "bonsai"

        # ============================================================
        # Physiological State Variables
        # ============================================================
        self.health = 100
        self.water = 100
        self.fertilizer = 30
        self.status = "alive"

        # ============================================================
        # Structural Attributes
        # ============================================================
        self.height = 1
        self.branches = 1
        self.curvature = 0
        self.inclination = 0

        # ============================================================
        # Species-Specific Parameters
        # ============================================================
        self.specie = random.choice(self.PLANT_SPECIES)
        self.waterMax = self.specie["water_max"]
        self.waterNeed = self.specie["water_need"]
        self.wired_min = self.specie["wire_min_year"]
        self.wired_max = self.specie["wire_max_year"]
        self.repotting_year = self.specie["repotting_year"]
        self.max_inclination = self.specie["max_inclination"]
        self.max_curvature = self.specie["max_curvature"]
        self.min_curvature = self.specie["min_curvature"]
        self.growth_rate = self.specie["growth_rate"]

        # ============================================================
        # Initial Stylistic Configuration
        # ============================================================
        initial_style = random.choice(
            ["Chokkan", "Moyogi", "Shakan", "Kengai", "Han-Kengai"]
        )

        if initial_style == "Chokkan":
            self.inclination = round(random.uniform(80, 100), 2)
            self.curvature = round(random.uniform(0, 2), 2)

        elif initial_style == "Moyogi":
            self.inclination = round(random.uniform(60, 120), 2)
            self.curvature = round(random.uniform(3, 5), 2)

        elif initial_style == "Shakan":
            self.inclination = (
                round(random.uniform(10, 60), 2)
                if random.random() < 0.5
                else round(random.uniform(120, 170), 2)
            )
            self.curvature = round(random.uniform(0, 2), 2)

        elif initial_style == "Kengai":
            self.inclination = round(random.uniform(180, 200), 2)
            self.curvature = round(random.uniform(3, 5), 2)

        elif initial_style == "Han-Kengai":
            self.inclination = (
                round(random.uniform(0, 10), 2)
                if random.random() < 0.5
                else round(random.uniform(170, 180), 2)
            )
            self.curvature = round(random.uniform(2, 3), 2)

        # ============================================================
        # Initial Morphological State
        # ============================================================
        self.height = round(random.uniform(5, 199), 2)
        self.original_height = self.height
        self.pot_size = self.height * 2 / 3
        self.branches = round(random.uniform(0, 5), 2)

        # Final classified style based on morphology
        self.style = self.classify_style()

        # ============================================================
        # Care and Maintenance Counters
        # ============================================================
        self.repotting_counter = 0
        self.pruning_counter = 0
        self.wire_counter = 0
        self.fertilizer_counter = 0

        self.wired = False
        self.daily_care = False
        self.score_caregiver = 0

        # ============================================================
        # Growth Model Parameters
        # ============================================================
        self.N = self.original_height
        self.r = 0.0005
        self.alpha = 1.2
        self.beta = 1.2
        self.sigma = 1.5
        self.K = 200

        self.steps = YEAR * YEAR_QUANTITY
        self.sizes = [self.N]

    def calculate_growth(self):
        """
        Updates bonsai height using a generalized logistic growth model.

        Growth follows a bounded nonlinear dynamic with carrying capacity K.
        Once the maximum size is reached, growth saturates.
        """
        if self.N >= self.K:
            self.N = self.K
            self.height = self.N
        else:
            dN_dt = self.r * self.N**self.alpha * (1 - (self.N / self.K)**self.beta)**self.sigma
            self.N += dN_dt  
            self.N = max(0, self.N)
            self.height = self.N
            self.sizes.append(self.N)

    # TODO: move this method to analysis/visualization module
    def plot_growth(self):
        plt.plot(range(self.steps + 1), self.sizes)
        plt.xlabel("Simulation steps")
        plt.ylabel("Bonsai height")
        plt.title("Bonsai growth trajectory")
        plt.show()

    def classify_style(self):
        """
        Classifies the bonsai into a traditional style based on
        inclination and curvature thresholds.

        Returns
        -------
        str
            One of the canonical bonsai styles or 'No style defined'.
        """
        
        if self.inclination >= 80 and self.inclination <= 100 and self.curvature <= 2:
            return "Chokkan"  
        elif self.inclination >= 60 and self.inclination <= 120 and self.curvature >= 3:
            return "Moyogi"   
        elif (self.inclination >= 10 and self.inclination <= 60 and self.curvature <= 2) or (self.inclination >= 120 and self.inclination <= 170 and self.curvature <= 2):
            return "Shakan"
        elif self.inclination >= 180 and self.curvature >= 3:
            return "Kengai"  
        elif (self.inclination >= 0 and self.inclination <= 10 and self.curvature <= 3 and self.curvature >= 2) or (self.inclination >= 170 and self.inclination <= 180 and self.curvature <= 3  and self.curvature >= 2):
            return "Han-Kengai"   
        else:
            return "No style defined"

    def adjust_to_center(self):
        """
        Adjusts bonsai inclination and curvature toward the ideal values
        of its current (or closest) bonsai style.

        This method represents a knowledgeable caregiver who understands
        the geometric targets of each bonsai style and applies wiring
        gradually, respecting physical constraints.
        """
        style = self.classify_style()

        # If the bonsai does not clearly belong to a known style,
        # find the closest style in geometric space (inclination, curvature)
        if style == "No style defined":

            styles = {
                "Chokkan": {"target_inclination": 90, "target_curvature": 0.5},
                "Moyogi": {"target_inclination": 90, "target_curvature": 4.5},
                "Shakan": {"targets_inclinations": [35, 145], "target_curvature": 1.5},
                "Kengai": {"target_inclination": 270, "target_curvature": 3.5},
                "Han-Kengai": {"targets_inclinations": [5, 175], "target_curvature": 2.5}
            }

            min_distance = float('inf')

            # Compute Euclidean distance to each style prototype
            for s, values in styles.items():

                # Styles with two valid inclination targets
                if 'targets_inclinations' in values:
                    t1, t2 = values['targets_inclinations']
                    d1 = abs(self.inclination - t1)
                    d2 = abs(self.inclination - t2)
                    target_inclination = t1 if d1 < d2 else t2
                    inclination_distance = min(d1, d2)
                else:
                    target_inclination = values["target_inclination"]
                    inclination_distance = abs(self.inclination - target_inclination)

                target_curvature = values["target_curvature"]
                curvature_distance = abs(self.curvature - target_curvature)

                # Combined geometric distance
                total_distance = (inclination_distance**2 + curvature_distance**2)**0.5

                if total_distance < min_distance:
                    min_distance = total_distance
                    best_target_inclination = target_inclination
                    best_target_curvature = target_curvature

            target_inclination = best_target_inclination
            target_curvature = best_target_curvature

        # If style is already known, directly assign its ideal targets
        elif style == "Chokkan":
            target_inclination, target_curvature = 90, 0.5
        elif style == "Moyogi":
            target_inclination, target_curvature = 90, 4.5
        elif style == "Shakan":
            target_inclination = 35 if abs(self.inclination - 35) < abs(self.inclination - 145) else 145
            target_curvature = 1.5
        elif style == "Kengai":
            target_inclination, target_curvature = 270, 3.5
        elif style == "Han-Kengai":
            target_inclination = 5 if abs(self.inclination - 5) < abs(self.inclination - 175) else 175
            target_curvature = 2.5

        # Maximum adjustment per step (biophysical constraint)
        max_inclination = 7
        max_curvature = 0.5

        # Gradual inclination correction
        if self.inclination < target_inclination:
            self.inclination = min(
                self.inclination + random.uniform(0, max_inclination),
                target_inclination
            )
        elif self.inclination > target_inclination:
            self.inclination = max(
                self.inclination - random.uniform(0, max_inclination),
                target_inclination
            )

        # Gradual curvature correction
        if self.curvature < target_curvature:
            self.curvature = min(
                self.curvature + random.uniform(0, max_curvature),
                target_curvature
            )
        elif self.curvature > target_curvature:
            self.curvature = max(
                self.curvature - random.uniform(0, max_curvature),
                target_curvature
            )

        # Wiring is applied during this step
        self.wired = True


    def adjust_to_center_no_knowledgment(self):
        """
        Adjusts inclination and curvature randomly, without any
        knowledge of bonsai styles.

        This represents an unskilled caregiver whose actions may
        improve or worsen the bonsai configuration.
        """
        max_inclination = 7
        max_curvature = 0.5

        self.inclination += random.uniform(-max_inclination, max_inclination)
        self.curvature += random.uniform(-max_curvature, max_curvature)

        self.wired = True


    def adjust_to_center_some_knowledgment(self, style_experience):
        """
        Adjusts inclination and curvature using partial knowledge
        of bonsai styles.

        Target values are scaled by the caregiver's experience
        level for each style, modeling imperfect learning.
        """
        style = self.classify_style()

        # Same geometric inference as the full-knowledge case
        if style == "No style defined":

            styles = {
                "Chokkan": {"target_inclination": 90, "target_curvature": 0.5},
                "Moyogi": {"target_inclination": 90, "target_curvature": 4.5},
                "Shakan": {"targets_inclinations": [35, 145], "target_curvature": 1.5},
                "Kengai": {"target_inclination": 270, "target_curvature": 3.5},
                "Han-Kengai": {"targets_inclinations": [5, 175], "target_curvature": 2.5}
            }

            min_distance = float('inf')

            for s, values in styles.items():
                if 'targets_inclinations' in values:
                    t1, t2 = values['targets_inclinations']
                    d1, d2 = abs(self.inclination - t1), abs(self.inclination - t2)
                    target_inclination = t1 if d1 < d2 else t2
                    inclination_distance = min(d1, d2)
                else:
                    target_inclination = values["target_inclination"]
                    inclination_distance = abs(self.inclination - target_inclination)

                target_curvature = values["target_curvature"]
                curvature_distance = abs(self.curvature - target_curvature)

                total_distance = (inclination_distance**2 + curvature_distance**2)**0.5

                if total_distance < min_distance:
                    min_distance = total_distance
                    best_target_inclination = target_inclination
                    best_target_curvature = target_curvature

            target_inclination = best_target_inclination
            target_curvature = best_target_curvature

        elif style == "Chokkan":
            target_inclination, target_curvature = 90, 0.5
        elif style == "Moyogi":
            target_inclination, target_curvature = 90, 4.5
        elif style == "Shakan":
            target_inclination = 35 if abs(self.inclination - 35) < abs(self.inclination - 145) else 145
            target_curvature = 1.5
        elif style == "Kengai":
            target_inclination, target_curvature = 270, 3.5
        elif style == "Han-Kengai":
            target_inclination = 5 if abs(self.inclination - 5) < abs(self.inclination - 175) else 175
            target_curvature = 2.5

        # Style-dependent experience scaling
        style_indices = {
            "Chokkan": 0,
            "Moyogi": 1,
            "Shakan": 2,
            "Kengai": 3,
            "Han-Kengai": 4,
            "No style defined": 5
        }

        experience = style_experience[style_indices.get(style, 5)]

        target_inclination *= experience
        target_curvature *= experience

        max_inclination = 7
        max_curvature = 0.5

        if self.inclination < target_inclination:
            self.inclination = min(
                self.inclination + random.uniform(0, max_inclination),
                target_inclination
            )
        elif self.inclination > target_inclination:
            self.inclination = max(
                self.inclination - random.uniform(0, max_inclination),
                target_inclination
            )

        if self.curvature < target_curvature:
            self.curvature = min(
                self.curvature + random.uniform(0, max_curvature),
                target_curvature
            )
        elif self.curvature > target_curvature:
            self.curvature = max(
                self.curvature - random.uniform(0, max_curvature),
                target_curvature
            )

        self.wired = True


    def decrease_health(self, amount = 5):
        """
        Decreases bonsai health due to environmental stress
        or lack of maintenance.
        """
        if(self.fertilizer == 0 
           or self.water < self.waterNeed 
           or self.repotting_counter >= (self.repotting_year * YEAR)
           or self.wire_counter >= (self.wired_max * YEAR)):
            
            self.health -= amount
            self.health = max(self.health, 0)

    def increase_health(self, amount = 5):
        """
        Increases the bonsai's health up to a maximum of 100.
        Represents recovery due to adequate care or favorable conditions.
        """
        self.health = min(self.health + amount, 100)

    def decrease_health_pruning(self, amount = 5):
        """
        Decreases health as a consequence of pruning.
        Models physiological stress caused by excessive or aggressive pruning.
        """
        self.health = max(self.health - amount, 0)
    
    def calculate_inclination(self):
        """
        Updates inclination as a function of wind intensity.
        """
        if self.model.wind_strength < 1:
            self.inclination += random.uniform(0, 0.01)
        elif self.model.wind_strength < 2:
            self.inclination += random.uniform(0.01, 0.05)
        elif self.model.wind_strength < 3:
            self.inclination -= random.uniform(0, 0.01)
        else:
            self.inclination -= random.uniform(0.01, 0.05)

        self.inclination = max(0, min(self.inclination, 360))

    def calculate_curvature(self):
        """
        Updates curvature as a function of light exposure.
        """ 
        if self.model.light_strength < 1:
            self.curvature += random.uniform(0, 0.001)
        elif self.model.light_strength < 2:
            self.curvature += random.uniform(0.001, 0.005)
        elif self.model.light_strength < 3:
            self.curvature -= random.uniform(0, 0.001)
        else:
            self.curvature -= random.uniform(0.001, 0.005)

        self.curvature = max(0, min(self.curvature, 5))

    def increase_height(self, centimeter = 0):
        """
        Increases bonsai height during the growing season.
        Growth depends on health, fertilizer availability, and seasonal conditions.
        """
        if self.model.season != 'winter' and self.health > GROWING_HEALTH_THRESHOLD:         
            self.height = min(self.height + (centimeter * self.growth_rate) + ((self.fertilizer * 0.10) * self.model.growth_season), 200)

    def decrease_height(self):
        """
        Resets height to its original value.
        Used to model drastic pruning or structural correction.
        """
        self.height = self.original_height
        self.N = self.original_height

    def increase_branches(self, branch = 0):
        """
        Increases the number of branches.
        Represents branch development due to growth or successful pruning strategies.
        """
        self.branches += branch
        
    def decrease_branches(self, branch = 0):
        """
        Decreases the number of branches.
        Represents pruning actions or branch loss.
        """
        self.branches -= branch

    def increase_water(self, water = 0):
        """
        Increases water level up to the species-specific maximum.
        Models watering actions performed by the caregiver.
        """
        self.water = min(self.water + water, self.waterMax)

    def decrease_water(self, water = 0):
        """
        Decreases water due to natural consumption.
        Water consumption scales with plant size and environmental factors.
        """
        self.water = max(self.water - ((water * self.model.water_consumption) + (self.height * 0.1)), 0)

    def increase_fertilizer(self, fertilizer = 0):
        """
        Increases fertilizer availability.
        Fertilizer contributes indirectly to growth and health recovery.
        """
        self.fertilizer = max(self.fertilizer + fertilizer, 30)

    def decrease_fertilizer(self, fertilizer = 0):
        """
        Decreases fertilizer over time, except during winter.
        Remaining fertilizer contributes to slight health recovery.
        """
        if self.model.season != "winter":
            self.fertilizer = max(self.fertilizer - fertilizer, 0)
            if self.fertilizer > 0:
                self.increase_health(1)

    def check_status(self):
        """
        Checks whether the bonsai is still alive.
        Death occurs if health or water reaches zero.
        """
        if self.health <= 0 or self.water <= 0:
            print(" O bonsai morreu: ", self.unique_id)
            print("Step: ", self.model.step_count)
            print("Saude: ", self.health)
            print("Agua: ", self.water)
            self.status = 'dead' 

    def unwire(self):
        """
        Removes wiring from the bonsai.
        Resets wire-related stress counters.
        """
        self.wired = False
        self.wire_counter = 0

    def repot(self):
        """
        Repots the bonsai if its size exceeds pot capacity.
        Updates pot size and resets repotting counter.
        """
        if self.height * 2 / 3 > self.pot_size:
            self.original_height = self.height
            self.pot_size = self.height * 2 / 3

        self.repotting_counter = 0

    def increase_counters(self):
        """
        Increments internal counters related to maintenance actions.
        Used to model delayed stress from lack of care.
        """
        self.repotting_counter += 1
        self.pruning_counter += 1
        self.fertilizer_counter += 1
        if self.wired:
            self.wire_counter += 1

    def caregiver_step(self):
        """
        Placeholder for caregiver-driven actions.
        BonsaiAgent itself does not make decisions.
        """
        pass

    def adjust_structure(self, strategy="full", style_experience=None):
        """
        Adjusts the bonsai structure according to the selected wiring strategy.

        Parameters
        ----------
        strategy : str
            Type of structural knowledge applied:
            - 'none'    : random or naive adjustment
            - 'partial' : adjustment biased by caregiver experience
            - 'full'    : optimal structural correction
        style_experience : array-like, optional
            Caregiver experience per style (used in partial knowledge).
        """
        if strategy == "none":
            self.adjust_to_center_no_knowledgment()
        elif strategy == "partial":
            self.adjust_to_center_some_knowledgment(style_experience)
        else:
            self.adjust_to_center()


    def bonsai_step(self):
        """
        Executes one simulation step for the bonsai agent.
        """
        if self.status != 'dead':
            self.increase_counters()
            self.decrease_fertilizer(DECREASING_FERTILIZE)
            self.decrease_health(DECREASING_HEALTH)
            self.decrease_water(DECREASING_WATER)

            self.calculate_growth()

            self.calculate_inclination()
            self.calculate_curvature()
            
            self.check_status()
            
