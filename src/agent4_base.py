# This resource is useful for Agent4

class Interaction:
    """An interaction is a tuple (action, outcome) with a valence"""
    def __init__(self, action, outcome, valence):
        self.action = action
        self.outcome = outcome
        self.valence = valence

    def key(self):
        """ The key to find this interaction in the dictinary is the string '<action><outcome>'. """
        return f"{self.action}{self.outcome}"

    def __str__(self):
        """ Print interaction in the form '<action><outcome:<valence>' for debug."""
        return f"{self.action}{self.outcome}:{self.valence}"

    def __eq__(self, other):
        """ Interactions are equal if they have the same key """
        return self.key() == other.key()


class Agent:
    """Creating our agent"""
    def __init__(self, _interactions):
        """ Initialize the dictionary of interactions"""
        self._interactions = {interaction.key(): interaction for interaction in _interactions}
        self._intended_interaction = self._interactions["00"]

    def action(self, _outcome):
        """ Tracing the previous cycle """
        previous_interaction = self._interactions[f"{self._intended_interaction.action}{_outcome}"]
        print(f"Action: {self._intended_interaction.action}, Prediction: {self._intended_interaction.outcome}, "
              f"Outcome: {_outcome}, Prediction_correct: {self._intended_interaction.outcome == _outcome}, "
              f"Valence: {previous_interaction.valence})")

        """ Computing the next interaction to try to enact """
        # TODO: Implement the agent's decision mechanism
        intended_action = 0
        # TODO: Implement the agent's prediction mechanism
        intended_outcome = 0
        # Memorize the intended interaction
        self._intended_interaction = self._interactions[f"{intended_action}{intended_outcome}"]
        return intended_action
