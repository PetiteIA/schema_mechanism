import pandas as pd
import numpy as np


class Interaction:
    """An interaction is a tuple (action, outcome) with a valence"""
    def __init__(self, action, outcome, valence):
        self._action = action
        self._outcome = outcome
        self._valence = valence

    def get_action(self):
        """Return the action"""
        return self._action

    def get_primitive_action(self):
        """Return the action for compatibility with CompositeInteraction"""
        return self._action

    def get_outcome(self):
        """Return the action"""
        return self._outcome

    def get_valence(self):
        """Return the action"""
        return self._valence

    def key(self):
        """ The key to find this interaction in the dictinary is the string '<action><outcome>'. """
        return f"{self._action}{self._outcome}"

    def pre_key(self):
        """Return the key. Used for compatibility with CompositeInteraction"""
        return self.key()

    def get_primitive_series(self):
        """"Return the key in a list"""
        return [self.key()]

    def __str__(self):
        """ Print interaction in the form '<action><outcome:<valence>' for debug."""
        return f"{self._action}{self._outcome}:{self._valence}"

    def __eq__(self, other):
        """ Interactions are equal if they have the same key """
        if isinstance(other, self.__class__):
            return self.key() == other.key()
        else:
            return False

    # def get_valence_series(self):
    #     """Retyrb the valence in a list """
    #     return [self._valence]


class CompositeInteraction:
    """A composite interaction is a tuple (pre_interaction, post_interaction) and a weight"""

    def __init__(self, pre_interaction, post_interaction):
        self.pre_interaction = pre_interaction
        self.post_interaction = post_interaction
        self.weight = 1
        self.isActivated = False

    def get_action(self):
        """Return the action of the pre interaction"""
        return self.key()
        # return self.pre_interaction.get_action()

    def get_primitive_action(self):
        """Return the primite action"""
        return self.pre_interaction.get_primitive_action()

    def get_valence(self):
        """Return the valence of the pre_interaction plus the valence of the post_interaction"""
        return self.pre_interaction.get_valence() + self.post_interaction.get_valence()

    def reinforce(self):
        """Increment the composite interaction's weight"""
        self.weight += 1

    def key(self):
        """ The key to find this interaction in the dictionary is the string '<pre_interaction><post_interaction>'. """
        return f"({self.pre_interaction.key()},{self.post_interaction.key()})"

    def pre_key(self):
        """Return the key of the pre_interaction"""
        return self.pre_interaction.pre_key()

    def get_primitive_series(self):
        """"Return the list of primitive keys"""
        return self.pre_interaction.get_primitive_series() + self.post_interaction.get_primitive_series()

    def __str__(self):
        """ Print the interaction in the Newick tree format (pre_interaction, post_interaction: valence) """
        return f"({self.pre_interaction}, {self.post_interaction}: {self.weight})"

    def __eq__(self, other):
        """ Interactions are equal if they have the same pre and post interactions """
        if isinstance(other, self.__class__):
            return (self.pre_interaction == other.pre_interaction) and (self.post_interaction == other.post_interaction)
        else:
            return False

    def get_valence_series(self):
        """"Return the list of valences of primitive interactions"""
        return self.pre_interaction.get_valence_series() + self.post_interaction.get_valence_series()


class Agent:
    def __init__(self, _interactions):
        """ Initialize our agent """
        self._interactions = {interaction.key(): interaction for interaction in _interactions}
        self._composite_interactions = {}
        self._intended_interaction = self._interactions["00"]
        self._last_interaction = None
        self._previous_interaction = None
        self._penultimate_interaction = None
        self._last_composite_interaction = None
        self._previous_composite_interaction = None
        # Create a dataframe of default primitive interactions
        default_interactions = [interaction for interaction in _interactions if interaction.get_outcome() == 0]
        data = {'proposed': [i.key() for i in default_interactions],
                'E(Vi)': [0.] * len(default_interactions),
                'action': [i.get_action() for i in default_interactions],
                'E(Va)': [0.] * len(default_interactions),
                'interaction': [i.key() for i in default_interactions],
                'weight': [0] * len(default_interactions)}
        self.primitive_df = pd.DataFrame(data)
        # Store the selection dataframe as a class attribute so we can display it in the notebook
        self.selection_df = None

    def action(self, _outcome):
        """Implement the agent's policy"""
        # Memorize the context
        self._previous_composite_interaction = self._last_composite_interaction
        self._penultimate_interaction = self._previous_interaction
        self._previous_interaction = self._last_interaction
        self._last_interaction = self._interactions[f"{self._intended_interaction.get_action()}{_outcome}"]

        # tracing the previous cycle
        print(
            f"Action: {self._intended_interaction.get_action()}, Prediction: {self._intended_interaction.get_outcome()}, "
            f"Outcome: {_outcome}, Prediction_correct: {self._intended_interaction.get_outcome() == _outcome}, "
            f"Valence: {self._last_interaction.get_valence()}")

        # Call the learning mechanism
        self.learn()

        # Calculate the proposed dataframe
        self.calculate_proposed_df()

        # Select the intended primitive interaction
        self.select_intended_interaction()

        return self._intended_interaction.get_action()

    def learn(self):
        """Learn the composite interactions"""
        # First level of composite interactions
        self._last_composite_interaction = self.learn_composite_interaction(self._previous_interaction,
                                                                            self._last_interaction)
        # Second level of composite interactions
        self.learn_composite_interaction(self._previous_composite_interaction, self._last_interaction)
        self.learn_composite_interaction(self._penultimate_interaction, self._last_composite_interaction)

    def learn_composite_interaction(self, pre_interaction, post_interaction):
        if pre_interaction is None:
            return None
        else:
            # Record or reinforce the first level composite interaction
            composite_interaction = CompositeInteraction(pre_interaction, post_interaction)
            if composite_interaction.key() not in self._composite_interactions:
                self._composite_interactions[composite_interaction.key()] = composite_interaction
                print(f"Learning {composite_interaction}")
                return composite_interaction
            else:
                self._composite_interactions[composite_interaction.key()].reinforce()
                print(f"Reinforcing {self._composite_interactions[composite_interaction.key()]}")
                # Retrieve the existing composite interaction
                return self._composite_interactions[composite_interaction.key()]

    def calculate_proposed_df(self):
        """Select the action that has the highest expected valence"""

        # The activated composite interactions
        activated_keys = [composite_interaction.key() for composite_interaction in self._composite_interactions.values()
                          if composite_interaction.pre_interaction == self._last_interaction or
                          composite_interaction.pre_interaction == self._last_composite_interaction]

        # Create the dataframe of sequences
        series_df = pd.DataFrame(columns=['proposed', 'weight', 'a_t', 'i_t', 'a_t+1', 'i_t+1'])
        for k in activated_keys:
            new_row = {'proposed': self._composite_interactions[k].post_interaction.key(),
                       'weight': self._composite_interactions[k].weight,
                       'a_t': self._composite_interactions[k].post_interaction.get_primitive_action()}
            if type(self._composite_interactions[k].post_interaction) == Interaction:
                new_row['i_t'] = self._composite_interactions[k].post_interaction.key()
            else:
                new_row['i_t'] = self._composite_interactions[k].post_interaction.pre_interaction.key()
                new_row['a_t+1'] = self._composite_interactions[k].post_interaction.post_interaction.get_primitive_action()
                new_row['i_t+1'] = self._composite_interactions[k].post_interaction.post_interaction.key()
            series_df = pd.concat([series_df, pd.DataFrame([new_row])], ignore_index=True)
        # print(series_df)

        # The probability P(it|at)
        total_by_i = series_df.groupby(["a_t", "i_t"], as_index=False)["weight"].sum().rename(columns={"weight": "i_weight"})
        total_by_a = series_df.groupby("a_t", as_index=False)["weight"].sum().rename(columns={"weight": "a_weight"})
        p_t_df = pd.merge(total_by_i, total_by_a, on="a_t")
        p_t_df['P(it|at)'] = p_t_df['i_weight'] / p_t_df['a_weight']
        # print(p_t_df[['a_t', 'i_t', 'P(it|at)']])

        # The probability P(it+1|it, at+1)
        series_filtered = series_df.dropna(subset=["i_t+1"])
        total_by_i = series_filtered.groupby(["i_t", "a_t+1", "i_t+1"], as_index=False)["weight"].sum().rename(columns={"weight": "t+1_weight"})
        total_by_a = series_filtered.groupby(["i_t", "a_t+1"], as_index=False)["weight"].sum().rename(columns={"weight": "t_weight"})
        p_t1_df = pd.merge(total_by_i, total_by_a, on=["i_t", "a_t+1"])
        p_t1_df['P(it+1|it, at+1)'] = p_t1_df['t+1_weight'] / p_t1_df['t_weight']
        # print(p_t1_df[['i_t', 'a_t+1', 'i_t+1', 'P(it+1|it, at+1)']])  # [['i_t', 'a_t+1', 'i_t+1', 'P(it+1|it, at+1)']]

        # Create the dataframe of proposed interactions
        data = {'proposed': [self._composite_interactions[k].post_interaction.key() for k in activated_keys],
                'E(Vi)': [0.] * len(activated_keys),
                'action': [self._composite_interactions[k].post_interaction.get_primitive_action() for k in
                           activated_keys],
                'E(Va)': [0.] * len(activated_keys),
                'weight': [self._composite_interactions[k].weight for k in activated_keys],
                'interaction': [self._composite_interactions[k].post_interaction.pre_key() for k in activated_keys]
                }
        expected_df = pd.DataFrame(data)
        # Add default interactions
        expected_df = pd.concat([self.primitive_df, expected_df], ignore_index=True)

        # Shorten the composite interactions whose post_interaction have a negative valence
        for i, k in expected_df["proposed"].items():
            if k in self._composite_interactions and self._composite_interactions[k].post_interaction.get_valence() < 0:
                expected_df.at[i, "proposed"] = self._composite_interactions[k].pre_interaction.key()

        # Remove the interactions that are the beginning of a longer interaction
        to_remove = set()
        for i, k1 in expected_df["proposed"].items():
            for j, k2 in expected_df["proposed"].items():
                if i not in to_remove and j not in to_remove and i != j:
                    if k1 in self._composite_interactions:
                        s1 = pd.Series(self._composite_interactions[k1].get_primitive_series())
                    else:
                        s1 = pd.Series(k1)
                    if k2 in self._composite_interactions:
                        s2 = pd.Series(self._composite_interactions[k2].get_primitive_series())
                    else:
                        s2 = pd.Series(k2)
                    if len(s1) <= len(s2):
                        if s1.equals(s2.iloc[:len(s1)]):
                            # print(f"Remove {s1.tolist()} from {i} to {j} weight {expected_df.at[i, 'weight']}")
                            to_remove.add(i)  # Mark the sequence to be removed
                            expected_df.at[j, "weight"] += expected_df.at[i, "weight"]
                        elif s2.equals(s1.iloc[:len(s2)]):
                            # print(f"Remove {s2.tolist()} from {j} to {i} weight {expected_df.at[j, 'weight']}")
                            to_remove.add(j)  # Mark the sequence to be removed
                            expected_df.at[i, "weight"] += expected_df.at[j, "weight"]
        expected_df = expected_df.drop(index=to_remove)
        # Compute the expected valence of interactions
        for i, k in expected_df["proposed"].items():
            if k in self._interactions:
                first_row = p_t_df.loc[p_t_df["i_t"] == k].head(1)  # ["P(it|at)"].values[0]
                p = first_row["P(it|at)"].values[0] if not first_row.empty else 0
                # print(f"E(vi) of {k} probability {p}")
                expected_df.at[i, "E(Vi)"] = p * self._interactions[k].get_valence()
            else:
                k1 = self._composite_interactions[k].pre_interaction.key()
                p1 = p_t_df.loc[p_t_df["i_t"] == k1].head(1)["P(it|at)"].values[0]
                k2 = self._composite_interactions[k].post_interaction.key()
                p2 = p_t1_df.loc[p_t1_df["i_t+1"] == k2].head(1)["P(it+1|it, at+1)"].values[0]
                expected_df.at[i, "E(Vi)"] = p1 * (self._interactions[k1].get_valence()
                                                   + p2 * self._interactions[k2].get_valence())
        # The sum expected valence per action
        expected_df["E(Va)"] = expected_df.groupby("action")["E(Vi)"].transform("sum")

        # Find the most probable outcome for each action
        max_weight_df = expected_df.loc[expected_df.groupby('action')['weight'].idxmax(), ['action', 'interaction']].reset_index(
            drop=True)
        max_weight_df.columns = ['action', 'intended']
        expected_df = expected_df.merge(max_weight_df, on='action')

        # Store the dataframe for printing
        self.selection_df = expected_df.copy()

    def select_intended_interaction(self):

        # The sum weight per action
        grouped_df = self.selection_df.groupby('action').agg({'weight': 'sum'}).reset_index()
        self.selection_df = self.selection_df.merge(grouped_df, on='action', suffixes=('', '_sum'))
        # self.selection_df["sum_weight"] = self.selection_df.groupby("action")["weight"].transform("sum")

        # Compute the proclivity
        self.selection_df['proclivity'] = self.selection_df["weight_sum"] * self.selection_df['E(Va)']

        # Select the action that has the highest proclivity
        max_index = self.selection_df['proclivity'].idxmax()
        intended_interaction_key = self.selection_df.loc[max_index, ['intended']].values[0]
        print(f"Intended max proclivity {intended_interaction_key}")
        self._intended_interaction = self._interactions[intended_interaction_key]

    def select_intended_interaction2(self):
        """Selects the intended interaction from the proposed dataframe"""
        # Find the first row that has the highest proclivity
        max_index = self.selection_df['E(Va)'].idxmax()
        intended_interaction_key = self.selection_df.loc[max_index, ['intended']].values[0]
        print(f"Intended Max E(Va) {intended_interaction_key}")
        self._intended_interaction = self._interactions[intended_interaction_key]


class Environment6:
    """ The grid """
    def __init__(self):
        """ Initialize the grid """
        self.grid = np.array([[1, 0, 0, 1]])
        self.position = 1

    def outcome(self, action):
        """Take the action and generate the next outcome """
        if action == 0:
            # Move left
            if self.position > 1:
                # No bump
                self.position -= 1
                self.grid[0, 3] = 1
                outcome = 0
            elif self.grid[0, 0] == 1:
                # First bump
                outcome = 1
                self.grid[0, 0] = 2
            else:
                # Subsequent bumps
                outcome = 0
        else:
            # Move right
            if self.position < 2:
                # No bump
                self.position += 1
                self.grid[0, 0] = 1
                outcome = 0
            elif self.grid[0, 3] == 1:
                # First bump
                outcome = 1
                self.grid[0, 3] = 2
            else:
                # Subsequent bumps
                outcome = 0
        return outcome

    def display(self):
        """Display the grid"""
        display = self.grid.copy()
        display[0, self.position] = 8
        print(display)


# Instanciate the agent in Environment6
pd.set_option('display.max_columns', 8)
interactions = [
    Interaction(0, 0, -1),
    Interaction(0, 1, 1),
    Interaction(1, 0, -1),
    Interaction(1, 1, 1)
]
a = Agent(interactions)
e = Environment6()

# Run the interaction loop
outcome = 0

if __name__ == '__main__':
    """ The main loop controlling the interaction of the agent with the environment """
    outcome = 0
    for step in range(30):
        print(f"\n*** STEP {step} ***\n")
        action = a.action(outcome)
        outcome = e.outcome(action)
        print(a.selection_df)
        e.display()
