import random
import numpy as np
import pickle
from mesa import Agent
from config.parameters import *


class CaregiverAgent(Agent):
    def __init__(self, unique_id, model, influence=0, bonsais = [], actions_per_day=3):
        """
        Initializes a caregiver agent responsible for managing one or more bonsai agents.

        The caregiver combines:
        - Personal stylistic preferences
        - Experience accumulated over time
        - Reinforcement learning (Q-learning)
        - Optional imitation of a master policy

        Parameters
        ----------
        unique_id : int
            Unique identifier of the caregiver agent.
        model : Model
            Reference to the Mesa model.
        influence : float
            Weight of the caregiver's influence over bonsai decisions.
        bonsais : list
            List of BonsaiAgent instances under this caregiver's responsibility.
        actions_per_day : int
            Maximum number of care actions per simulation step.
        """
        super().__init__(unique_id, model)
        self.tipo = "cuidador"
        self.influence = influence
        self.bonsais = bonsais
        self.actions_per_day = actions_per_day
        self.experience_total_per_year = []
        self.experience_per_year_by_style = []

        self.style_experience = np.random.uniform(0, 1, 6)
        self.personal_preference = np.random.uniform(0, 1, 6)
        self.bias = random.uniform(-1, 1)
        self.previous_bonsais = {bonsai.unique_id: (bonsai.classify_style(), bonsai.health) for bonsai in self.bonsais}

        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.01
        self.q_table = {}

        with open("q_table_and_epsilon.pkl", "rb") as f:
            data_loaded = pickle.load(f)
            self.q_table = data_loaded['q_table']
            self.epsilon = data_loaded['epsilon']
        
        with open("q_table_30_years.pkl", "rb") as f:
            data_loaded = pickle.load(f)
            self.q_table_master = data_loaded['q_table']
            self.epsilon_master = data_loaded['epsilon']

        self.actions = ['Fertilization', 'Pruning', 'Wiring', 'Watering', 'Unwiring', 'Repotting']
        self.reward_per_step = []

    def bonsai_step(self):
        """
        Placeholder method.

        Can be used if, in the future, caregivers need to react
        to bonsai-driven events rather than acting every model step.
        """
        pass

    def get_style_index(self, style):
        """
        Maps a bonsai style name to a fixed index.

        This index is used to access:
        - Style experience vector
        - Personal preference vector

        Returns
        -------
        int
            Index corresponding to the style, or -1 if undefined.
        """
        style_indices = {
        "Chokkan": 0,
        "Moyogi": 1,
        "Shakan": 2,
        "Kengai": 3,
        "Han-Kengai": 4,
        "No style defined": 5
        }

        return style_indices.get(style, -1)

    def evaluate_bonsai(self, bonsai):
        """
        Computes a subjective score for a bonsai based on:

        - Caregiver's personal preference for the bonsai style
        - Caregiver's experience with that style
        - Current health of the bonsai

        The score represents how "valuable" the bonsai is
        from the caregiver's perspective.
        """
        style = bonsai.classify_style()
        style_index = self.get_style_index(style)
        preference_score = self.personal_preference[style_index]
        experience_factor = self.style_experience[style_index]

        final_score = preference_score * experience_factor * (bonsai.health** 0.5)

        bonsai.score_caregiver = final_score

    def categorize_value(self, value, threshold, max_value):
        """
        Discretizes a continuous variable into three categories:
        0 = low
        1 = medium
        2 = high

        Used to reduce state space complexity for Q-learning.
        """
        if value < threshold:
            return 0
        elif value < max_value:
            return 1
        else:
            return 2

    def round_state(self, bonsai):
        """
        Converts the full bonsai state into a discrete representation.

        This discretized state is used as a key in the Q-table
        and includes health, water, season, wiring status, and counters.
        """
        health = self.categorize_value(bonsai.health, 50, 80)  # 0 = Baixo, 1 = Médio, 2 = Alto
        water = self.categorize_value(bonsai.water, 30, 70)  # 0 = Baixo, 1 = Médio, 2 = Alto
        actual_season = self.model.season
        wired = bonsai.wired

        return (health, water, self.categorize_value(bonsai.fertilizer_counter, 10, 30), actual_season, wired, self.categorize_value(bonsai.repotting_counter, (bonsai.repotting_year * PERCENT_OF_DAYS_TO_PRUNE_REPOT * YEAR), 365), self.categorize_value(bonsai.wire_counter, (bonsai.wired_min * YEAR), 365), self.categorize_value(bonsai.pruning_counter, (YEAR * PERCENT_OF_DAYS_TO_PRUNE_REPOT), 365))

    def choose_action(self, bonsai):
        """
        Selects an action using an epsilon-greedy policy.

        Priority-based overrides:
        - Watering if water is critically low
        - Fertilization if fertilizer is depleted

        Otherwise, chooses between exploration and exploitation
        based on the current epsilon value.
        """
        state = self.round_state(bonsai)

        if bonsai.water < 40:
            return 'Watering'
        
        if bonsai.fertilizer == 0:
            return 'Fertilization'
        
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)

        return action

    def choose_action_master(self, bonsai):
        """
        Selects an action based on a pre-trained 'master' Q-table.

        This represents expert knowledge accumulated over
        long-term simulations (e.g., 30 years).
        """
        state = self.round_state(bonsai)

        if bonsai.water < 40:
            return 'Watering'
        
        if bonsai.fertilizer == 0:
            return 'Fertilization'
        
        if state not in self.q_table_master:
            self.q_table_master[state] = {action: 0.0 for action in self.actions}
        
        if random.uniform(0, 1) < self.epsilon_master:
            action = random.choice(self.actions)
        else:
            action = max(self.q_table_master[state], key=self.q_table_master[state].get)

        return action

    def choose_action_no_knowledgement(self, bonsai):
        """
        Selects actions randomly, except for basic survival rules.

        Used when the model operates without learning or prior knowledge.
        """
        if bonsai.water < 40:
            return 'Watering'
        
        if bonsai.fertilizer == 0:
            return 'Fertilization'

        action = random.choice(self.actions)

        return action

    def update_q_table(self, state_before, action, reward, state_after,  master_action = None):
        """
        Updates the Q-table using standard Q-learning or
        imitation learning from a master policy.

        If a master action is provided, the caregiver partially
        aligns its Q-values with the master Q-table.
        """
        state = state_before
        next_state = state_after

        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}

        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.actions}

        if state not in self.q_table_master:
            self.q_table_master[state] = {action: 0.0 for action in self.actions}

        if master_action not in self.q_table_master[state]:
            self.q_table_master[state][master_action] = 0.0

        if next_state not in self.q_table_master:
            self.q_table_master[next_state] = {action: 0.0 for action in self.actions}

        if master_action is not None:
            q_value = self.q_table_master[state][master_action]
            #self.q_table[state][master_action] = self.q_table_master[state][master_action]
            self.q_table[state][master_action] = (1 - 0.7) * self.q_table[state][master_action] + 0.7 * self.q_table_master[state][master_action]
            #self.q_table[state][master_action] = (1 - self.alpha) * q_value + self.alpha * (reward + self.gamma * self.q_table_master[next_state][master_action])
        else:
            best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
            q_value = self.q_table[state][action]
            self.q_table[state][action] = q_value + self.alpha * (reward + self.gamma * self.q_table[next_state][best_next_action] - q_value)

    def caregiver_step(self):
        """
        Main decision loop executed at each simulation step.

        Behavior depends on the model's knowledge level:
        0 - Random actions (no learning)
        1 - Pure reinforcement learning
        2 - Hybrid learning (RL + master imitation)

        Also handles:
        - Reward accumulation
        - Epsilon decay
        - Annual experience updates
        """
        if self.model.knowledgement == 0:
            for bonsai in self.bonsais:
                if bonsai.status == 'alive':
                    action = self.choose_action_no_knowledgement(bonsai)
                    self.execute_action(bonsai, action)

        elif self.model.knowledgement == 1:

            total_reward = 0
        
            for bonsai in self.bonsais:

                if bonsai.status == 'alive':

                    state_before = self.round_state(bonsai) 

                    action = self.choose_action(bonsai)

                    reward = self.calculate_reward(bonsai, action)

                    total_reward += reward
                    
                    self.execute_action(bonsai, action)

                    state_after = self.round_state(bonsai)

                    self.update_q_table(state_before, action, reward, state_after)

                    self.evaluate_bonsai(bonsai)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.reward_per_step.append(total_reward)
        
        elif self.model.knowledgement == 2:

            total_reward = 0
        
            for bonsai in self.bonsais:

                if bonsai.status == 'alive':

                    if bonsai.health <= 50:

                        state_before = self.round_state(bonsai) 

                        action = self.choose_action(bonsai)

                        action_master = self.choose_action_master(bonsai)

                        reward = self.calculate_reward(bonsai, action_master)

                        self.execute_action(bonsai, action_master)

                        state_after = self.round_state(bonsai)

                        self.update_q_table(state_before, action, reward, state_after, master_action=action_master)

                    else:
                        state_before = self.round_state(bonsai) 

                        action = self.choose_action(bonsai)

                        reward = self.calculate_reward(bonsai, action)
                        
                        self.execute_action(bonsai, action)

                        state_after = self.round_state(bonsai)

                        self.update_q_table(state_before, action, reward, state_after)

                        self.evaluate_bonsai(bonsai)

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        if self.model.step_count % YEAR == 0:

            self.experience_total_per_year.append(self.style_experience.sum())

            self.experience_per_year_by_style.append(self.style_experience.copy())

            self.update_experience(self.previous_bonsais, {bonsai.unique_id: (bonsai.classify_style(), bonsai.health) for bonsai in self.bonsais})

            self.previous_bonsais = {bonsai.unique_id: (bonsai.classify_style(), bonsai.health) for bonsai in self.bonsais}

    def save_q_table(self):
        """
        Persists the caregiver's Q-table to disk
        for later reuse or analysis.
        """
        with open("q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)

    def calculate_reward(self, bonsai, action):
        """
        Computes the reward signal for a given action.

        Rewards are domain-specific and reflect bonsai care quality,
        seasonal correctness, and long-term health impact.
        """
        if action == 'Watering':
            if bonsai.water < bonsai.waterMax * 0.7:
                if bonsai.water < bonsai.waterNeed:
                    return 20
                else:
                    return 10
            else:
                return 0
        elif action == 'Fertilization':
            if bonsai.fertilizer == 0:
                return -10
            elif bonsai.fertilizer < 5:
                return 15
            elif bonsai.fertilizer < 15:
                return 5
            else:
                return -10
        elif action == 'Pruning':
            if bonsai.specie['prune_season'] == self.model.season and bonsai.health > PRUNING_HEALTH_THRESHOLD:
                if bonsai.pruning_counter > (YEAR * PERCENT_OF_DAYS_TO_PRUNE_REPOT):
                    return 15
                else:
                    return 10
            else:
                return -10
        elif action == 'Wiring':
            if bonsai.wired:  
                return -20
            else:
                if bonsai.specie['prune_season'] == self.model.season:
                    return 20
                else:
                    return -10
        elif action == 'Unwiring':
            if not bonsai.wired:
                return -20
            elif bonsai.wire_counter >= (bonsai.wired_max * YEAR):
                return -10
            elif bonsai.wire_counter >= (bonsai.wired_min * YEAR):
                return 20
            else:
                return -10
        elif action == 'Repotting':
            if bonsai.repotting_counter >= (bonsai.repotting_year * PERCENT_OF_DAYS_TO_PRUNE_REPOT * YEAR):
                if bonsai.specie['prune_season'] == self.model.season:
                    if bonsai.repotting_counter <= (bonsai.repotting_year * YEAR):
                        return 20
                    else:
                        return 10
                else:
                    return -20
            else:
                return -20

        return 0

    def execute_action(self, bonsai, action):
        """
        Applies the chosen action to the bonsai.

        This method encapsulates all side effects:
        - Health changes
        - Counters reset
        - Structural modifications
        - Style reclassification
        """
        if action == 'Watering':
            bonsai.increase_water(INCREASE_WATER_WATERING)
        elif action == 'Fertilization':
            if bonsai.fertilizer_counter > 25:
                bonsai.increase_health(HEALTH_WELLCARE)
                bonsai.fertilizer = 0
                bonsai.fertilizer_counter = 0
                bonsai.increase_fertilizer(INCREASE_FERTILIZER)
            elif bonsai.fertilizer_counter > 15:
                bonsai.fertilizer = 0
                bonsai.fertilizer_counter = 0
                bonsai.increase_fertilizer(INCREASE_FERTILIZER)
            else:
                bonsai.fertilizer = 0
                bonsai.fertilizer_counter = 0
                bonsai.increase_fertilizer(INCREASE_FERTILIZER)
                bonsai.decrease_health_pruning(HEALTH_WELLCARE)
        elif action == 'Pruning':
            if bonsai.health > PRUNING_HEALTH_THRESHOLD and bonsai.specie['prune_season'] == self.model.season:
                if bonsai.pruning_counter > (YEAR * PERCENT_OF_DAYS_TO_PRUNE_REPOT):
                    bonsai.increase_health(HEALTH_WELLCARE)
                bonsai.pruning_counter = 0
                bonsai.decrease_height()
                bonsai.decrease_branches(PRUNING_DECREASE_BRANCHE)
            else:
                bonsai.decrease_health_pruning(HEALTH_WELLCARE)
                bonsai.pruning_counter = 0
                bonsai.decrease_height()
                bonsai.decrease_branches(PRUNING_DECREASE_BRANCHE)
        elif action == 'Wiring':
            if not bonsai.wired:
                if bonsai.specie['prune_season'] == self.model.season:
                    bonsai.adjust_structure(
                        strategy=self.model.wiring_strategy,
                        style_experience=self.style_experience
                    )
                else:
                    bonsai.decrease_health_pruning(HEALTH_WELLCARE)
        elif action == 'Unwiring':
            if not bonsai.wired:
                return
            elif bonsai.wire_counter >= (bonsai.wired_max * YEAR):
                bonsai.decrease_health_pruning(HEALTH_WELLCARE)
                bonsai.unwire()
            elif bonsai.wire_counter >= (bonsai.wired_min * YEAR):
                bonsai.unwire()
            else:
                bonsai.decrease_health_pruning(HEALTH_WELLCARE)
        elif action == 'Repotting':
            if bonsai.repotting_counter >= (bonsai.repotting_year * PERCENT_OF_DAYS_TO_PRUNE_REPOT * YEAR) and bonsai.specie['prune_season'] == self.model.season:
                if bonsai.repotting_counter < (bonsai.repotting_year * YEAR):
                    bonsai.increase_health(HEALTH_WELLCARE)
                    bonsai.repot()
                else:
                    bonsai.decrease_health_pruning(HEALTH_WELLCARE)
                    bonsai.repot()
            else:
                bonsai.decrease_health_pruning(HEALTH_WELLCARE)
        bonsai.style = bonsai.classify_style()

    def print_q_table(self):
        """
        Prints the caregiver's Q-table in a human-readable format.

        Useful for debugging and qualitative analysis of learning.
        """
        print(f"\nCaregiver Q-table {self.unique_id}:")
        for state in self.q_table:
            print(f"State: {state} -> Actions e Q-values:")
            for action in self.q_table[state]:
                print(f"  Action: {action} | Q-value: {self.q_table[state][action]}")

    def update_experience(self, previous_bonsais, current_bonsais, learning_rate=0.05):
        """
        Updates caregiver experience per style based on bonsai outcomes.

        Experience increases if:
        - Health improves
        - Style remains stable

        Experience decreases if:
        - Health deteriorates
        - Style changes unexpectedly
        """
        for bonsai_id, (old_style, old_health) in previous_bonsais.items():
            if bonsai_id in current_bonsais:

                new_style, new_health = current_bonsais[bonsai_id]
                style_index = self.get_style_index(old_style)

                if new_style != old_style:
                    self.style_experience[style_index] -= learning_rate
                else:
                    if new_health >= old_health:
                        self.style_experience[style_index] += learning_rate
                    else:
                        self.style_experience[style_index] -= learning_rate * 0.5  
                
                self.style_experience[style_index] = np.clip(self.style_experience[style_index], 0, 1)
