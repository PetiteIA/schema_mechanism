# The solution to the notebook Agent8.ipynb

import pandas as pd
import numpy as np


class Interaction:
    """An interaction is a tuple (action, outcome) with a valence"""
    def __init__(self, _action, _outcome, _valence):
        self._action = _action
        self._outcome = _outcome
        self._valence = _valence

    def get_action(self):
        """Return the action"""
        return self._action

    def get_decision(self):
        """Return the decision key"""
        return f"a{self._action}"

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
    """A composite interaction is a tuple (pre_interaction, decision, post_interaction) and a weight"""
    def __init__(self, pre_interaction, decision, post_interaction):
        self.pre_interaction = pre_interaction
        self.decision = decision
        self.post_interaction = post_interaction
        self.weight = 1
        self.isActivated = False

    def get_decision(self):
        """Return the sequence of decisions"""
        return f"{self.pre_interaction.key()}{self.post_interaction.get_decision()}"
        return f"{self.pre_interaction.get_decision()}{self.post_interaction.get_decision()}"

    def get_primitive_action(self):
        """Return the primitive action"""
        return self.pre_interaction.get_primitive_action()

    def get_valence(self):
        """Return the valence of the pre_interaction plus the valence of the post_interaction"""
        return self.pre_interaction.get_valence() + self.post_interaction.get_valence()

    def reinforce(self):
        """Increment the composite interaction's weight"""
        self.weight += 1

    def key(self):
        """ The key to find this interaction in the dictionary is the string
        '<pre_interaction>,<decision>,<post_interaction>'. """
        return f"({self.pre_interaction.key()},{self.decision},{self.post_interaction.key()})"

    def pre_key(self):
        """Return the key of the pre_interaction"""
        return self.pre_interaction.pre_key()

    def __str__(self):
        """ Print the interaction in the Newick tree format (pre_interaction, post_interaction: valence) """
        return f"({self.pre_interaction}, {self.decision}, {self.post_interaction}: {self.weight})"

    def __eq__(self, other):
        """ Interactions are equal if they have the same keys """
        if isinstance(other, self.__class__):
            return self.key() == other.key()
        else:
            return False


class Agent:
    def __init__(self, _interactions):
        """ Initialize our agent """
        self._interactions = {interaction.key(): interaction for interaction in _interactions}
        self._composite_interactions = {}
        self._intended_interaction = self._interactions["00"]
        self._decision = None  # "0"
        self._last_interaction = None
        self._previous_interaction = None
        self._penultimate_interaction = None
        self._last_composite_interaction = None
        self._previous_composite_interaction = None
        # Create a dataframe of default primitive interactions
        default_interactions = [interaction for interaction in _interactions if interaction.get_outcome() == 0]
        data = {'activated': [np.nan] * len(default_interactions),
                'weight': [0] * len(default_interactions),
                'action': [i.get_primitive_action() for i in default_interactions],
                'interaction': [i.key() for i in default_interactions],
                'valence': [i.get_valence() for i in default_interactions],
                'decision': [i.get_decision() for i in default_interactions],
                'proclivity': [0] * len(default_interactions)}
        self.primitive_df = pd.DataFrame(data)
        # Store the selection dataframe as a class attribute so we can display it in the notebook
        self.proposed_df = None

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

        # Create the proposed dataframe
        self.create_proposed_df()
        self.aggregate_propositions()

        # Select the intended primitive interaction
        self.decide()

        return self._intended_interaction.get_action()

    def learn(self):
        """Learn the composite interactions"""
        # First level of composite interactions
        self._last_composite_interaction = self.learn_composite_interaction(
            self._previous_interaction, self._decision, self._last_interaction)
        # Second level of composite interactions
        self.learn_composite_interaction(self._previous_composite_interaction, self._decision, self._last_interaction)
        self.learn_composite_interaction(self._penultimate_interaction, self._decision, self._last_composite_interaction)

    def learn_composite_interaction(self, pre_interaction, decision, post_interaction):
        """Record or reinforce the composite interaction made of (pre_interaction, post_interaction)"""
        if pre_interaction is None:
            return None
        else:
            # If the pre interaction exist
            composite_interaction = CompositeInteraction(pre_interaction, decision, post_interaction)
            if composite_interaction.key() not in self._composite_interactions:
                # Add the composite interaction to memory
                self._composite_interactions[composite_interaction.key()] = composite_interaction
                print(f"Learning {composite_interaction}")
                return composite_interaction
            else:
                # Reinforce the existing composite interaction and return it
                self._composite_interactions[composite_interaction.key()].reinforce()
                print(f"Reinforcing {self._composite_interactions[composite_interaction.key()]}")
                return self._composite_interactions[composite_interaction.key()]

    def create_proposed_df(self):
        """Create the proposed dataframe from the activated interactions"""
        # The list of activated interaction that match the current context
        activated_keys = [composite_interaction.key() for composite_interaction in
                          self._composite_interactions.values()
                          if composite_interaction.pre_interaction == self._last_interaction or
                          composite_interaction.pre_interaction == self._last_composite_interaction]
        data = {'activated': activated_keys,
                'weight': [self._composite_interactions[k].weight for k in activated_keys],
                'action': [self._composite_interactions[k].post_interaction.get_primitive_action() for k in activated_keys],
                'interaction': [self._composite_interactions[k].post_interaction.pre_key() for k in activated_keys],
                'valence': [self._composite_interactions[k].post_interaction.get_valence() for k in activated_keys],
                'decision': [self._composite_interactions[k].post_interaction.get_decision() for k in activated_keys],
                }
        activated_df = pd.DataFrame(data)

        # Create the selection dataframe from the primitive and the activated dataframes
        self.proposed_df = pd.concat([self.primitive_df, activated_df], ignore_index=True)

        # # Compute the proclivity for each proposition
        self.proposed_df['proclivity'] = self.proposed_df['weight'] * self.proposed_df['valence']

    def aggregate_propositions(self):
        """Aggregate the proclivity"""
        # Compute the proclivity for each action
        grouped_df = self.proposed_df.groupby('decision').agg({'proclivity': 'sum'}).reset_index()
        self.proposed_df = self.proposed_df.merge(grouped_df, on='decision', suffixes=('', '_agg'))
        # Sort by descending order of proclivity
        self.proposed_df = self.proposed_df.sort_values(by=['proclivity_agg', 'decision'], ascending=[False, False])

        # Find the most probable primitive interaction for each action
        max_weight_df = self.proposed_df.loc[self.proposed_df.groupby('decision')['weight'].idxmax(), ['decision', 'interaction']].reset_index(
            drop=True)
        max_weight_df.columns = ['decision', 'intended']
        self.proposed_df = self.proposed_df.merge(max_weight_df, on='decision')

    def decide(self):
        """Selects the intended interaction from the proposed dataframe"""
        # Find the first row that has the highest proclivity
        max_index = self.proposed_df['proclivity_agg'].idxmax()
        self._decision = self.proposed_df.loc[max_index, ['decision']].values[0]
        # Find the intended interaction corresponding to the action that has the highest proclivity
        intended_interaction_key = self.proposed_df.loc[max_index, ['intended']].values[0]
        self._intended_interaction = self._interactions[intended_interaction_key]
        print(f"Decision {self._decision}, Intended {self._intended_interaction}")


class Environment7:
    """ The grid """
    def __init__(self):
        """ Initialize the grid and the agent's pose """
        self.grid = np.array([[1, 0, 0, 1]])
        self.position = 1
        self.direction = 0

    def outcome(self, _action):
        """Take the action and generate the next outcome """
        if _action == 0:
            # Move forward
            if self.direction == 0:
                # Move to the left
                if self.position > 1:
                    # No bump
                    self.position -= 1
                    self.grid[0, 3] = 1
                    _outcome = 0
                elif self.grid[0, 0] == 1:
                    # First bump
                    _outcome = 1
                    self.grid[0, 0] = 2
                else:
                    # Subsequent bumps
                    _outcome = 0
            else:
                # Move to the right
                if self.position < 2:
                    # No bump
                    self.position += 1
                    self.grid[0, 0] = 1
                    _outcome = 0
                elif self.grid[0, 3] == 1:
                    # First bump
                    _outcome = 1
                    self.grid[0, 3] = 2
                else:
                    # Subsequent bumps
                    _outcome = 0
        else:
            # Turn 180°
            _outcome = 0
            if self.direction == 0:
                self.direction = 1
            else:
                self.direction = 0
        return _outcome

    def display(self):
        """Display the grid"""
        int_to_char = {0: '-', 1: '|', 2: '*'}
        display_array = np.vectorize(int_to_char.get)(self.grid)
        if self.direction == 0:
            # Display agent to the left
            display_array[0, self.position] = "<"
        else:
            # Display agent to the right
            display_array[0, self.position] = ">"
        print(f"Environment: {' '.join(display_array[0])}")


# Instantiate the agent in Environment6
pd.set_option('display.max_columns', 10)
interactions = [
    Interaction(0, 0, -1),
    Interaction(0, 1, 1),
    Interaction(1, 0, -1),
    Interaction(1, 1, 1)
]
a = Agent(interactions)
e = Environment7()

# Run the interaction loop
outcome = 0

if __name__ == '__main__':
    """ The main loop controlling the interaction of the agent with the environment """
    outcome = 0
    for step in range(30):
        print(f"\n*** STEP {step} ***\n")
        action = a.action(outcome)
        outcome = e.outcome(action)
        print(a.proposed_df)
        e.display()
