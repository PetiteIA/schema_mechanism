# The solution of Notebook Agent6.ipynb

import pandas as pd


class Interaction:
    """An interaction is a tuple (action, outcome) with a valence"""
    def __init__(self, action, outcome, valence):
        self._action = action
        self._outcome = outcome
        self._valence = valence

    def get_action(self):
        """Return the action"""
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
        return self.key()

    def __str__(self):
        """ Print interaction in the form '<action><outcome:<valence>' for debug."""
        return f"{self._action}{self._outcome}:{self._valence}"

    def __eq__(self, other):
        """ Interactions are equal if they have the same key """
        if isinstance(other, self.__class__):
            return self.key() == other.key()
        else:
            return False


class CompositeInteraction:
    """A composite interaction is a tuple (pre_interaction, post_interaction) and a weight"""

    def __init__(self, pre_interaction, post_interaction):
        self.pre_interaction = pre_interaction
        self.post_interaction = post_interaction
        self.weight = 1
        self.isActivated = False

    def get_action(self):
        """Return the action of the post interaction"""
        return self.post_interaction.get_action()

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

    def __str__(self):
        """ Print the interaction in the Newick tree format (pre_interaction, post_interaction: valence) """
        return f"({self.pre_interaction}, {self.post_interaction}: {self.weight})"

    def __eq__(self, other):
        """ Interactions are equal if they have the same pre and post interactions """
        if isinstance(other, self.__class__):
            return (self.pre_interaction == other.pre_interaction) and (self.post_interaction == other.post_interaction)
        else:
            return False


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
        data = {'post_action': [i.get_action() for i in default_interactions],
                'weight': [0] * len(default_interactions),
                'proclivity': [0] * len(default_interactions),
                'post_interaction': [i.key() for i in default_interactions]}
        self.primitive_df = pd.DataFrame(data)
        # Store the selection dataframe as a class attribute so we can display it in the notebook
        self.selection_df = None

    def action(self, _outcome):
        """Implement the agent's policy"""
        # tracing the previous cycle
        self._previous_composite_interaction = self._last_composite_interaction
        self._penultimate_interaction = self._previous_interaction
        self._previous_interaction = self._last_interaction
        self._last_interaction = self._interactions[f"{self._intended_interaction.get_action()}{_outcome}"]
        print(
            f"Action: {self._intended_interaction.get_action()}, Prediction: {self._intended_interaction.get_outcome()}, "
            f"Outcome: {_outcome}, Prediction_correct: {self._intended_interaction.get_outcome() == _outcome}, "
            f"Valence: {self._last_interaction.get_valence()}")

        # Call the learning mechanism
        self.learn()

        # Create a dataframe from the activated composite interactions
        activated_keys = [composite_interaction.key() for composite_interaction in self._composite_interactions.values()
                          if composite_interaction.pre_interaction == self._last_interaction or
                          composite_interaction.pre_interaction == self._last_composite_interaction]
        data = {'composite': activated_keys,
                'weight': [self._composite_interactions[k].weight for k in activated_keys],
                'post_valence': [self._composite_interactions[k].post_interaction.get_valence() for k in
                                 activated_keys],
                'post_action': [self._composite_interactions[k].post_interaction.get_action() for k in activated_keys],
                'post_interaction': [self._composite_interactions[k].post_interaction.pre_key() for k in activated_keys]
                }
        activated_df = pd.DataFrame(data)

        # Create the selection dataframe from the primitive and the activated dataframes
        df = pd.concat([self.primitive_df, activated_df], ignore_index=True)

        # Compute the proclivity for each action
        df['proclivity'] = df['weight'] * df['post_valence']
        grouped_df = df.groupby('post_action').agg({'proclivity': 'sum'}).reset_index()
        df = df.merge(grouped_df, on='post_action', suffixes=('', '_sum'))

        # Find the most probable outcome for each action
        max_weight_df = df.loc[
            df.groupby('post_action')['weight'].idxmax(), ['post_action', 'post_interaction']].reset_index(drop=True)
        max_weight_df.columns = ['post_action', 'intended']
        df = df.merge(max_weight_df, on='post_action')

        # Find the first row that has the highest proclivity
        max_index = df['proclivity_sum'].idxmax()
        intended_interaction_key = df.loc[max_index, ['intended']].values[0]
        self._intended_interaction = self._interactions[intended_interaction_key]
        print("Intended", self._intended_interaction)

        # Store the selection dataframe for printing
        self.selection_df = df.copy()

        return self._intended_interaction.get_action()

    def learn(self):
        # Recording previous composite interaction
        if self._previous_interaction is not None:
            # Record or reinforce the first level composite interaction
            composite_interaction = CompositeInteraction(self._previous_interaction, self._last_interaction)
            if composite_interaction.key() not in self._composite_interactions:
                self._composite_interactions[composite_interaction.key()] = composite_interaction
                print(f"Learning {composite_interaction}")
                self._last_composite_interaction = composite_interaction
            else:
                self._composite_interactions[composite_interaction.key()].reinforce()
                print(f"Reinforcing {self._composite_interactions[composite_interaction.key()]}")
                self._last_composite_interaction = self._composite_interactions[composite_interaction.key()]
            # Record or reinforce the second level composite interaction
            if self._previous_composite_interaction is not None:
                composite_interaction_2 = CompositeInteraction(self._previous_composite_interaction, self._last_interaction)
                if composite_interaction_2.key() not in self._composite_interactions:
                    self._composite_interactions[composite_interaction_2.key()] = composite_interaction_2
                    print(f"Learning {composite_interaction_2}")
                else:
                    self._composite_interactions[composite_interaction_2.key()].reinforce()
                    print(f"Reinforcing {self._composite_interactions[composite_interaction_2.key()]}")


class Environment5:
    """ Environment5 """
    def __init__(self):
        """ Initializing Environment4 """
        self._previous_action = 0
        self._last_action = 0

    def outcome(self, _action):
        """Take the action and generate the next outcome """
        if _action == self._last_action and _action == self._previous_action:
            # If same action during the last 3 steps then outcome 0
            _outcome = 0
        else:
            # If different action then outcome 1
            _outcome = 1
        self._previous_action = self._last_action
        self._last_action = _action
        return _outcome


# Instanciate a new agent
pd.set_option('display.max_columns', 8)

interactions = [
    Interaction(0, 0, -1),
    Interaction(0, 1, 1),
    Interaction(1, 0, -1),
    Interaction(1, 1, 1),
]
a = Agent(interactions)
e = Environment5()

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
