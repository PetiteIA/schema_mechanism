import pandas as pd
from agent5_base import CompositeInteraction, Agent


class Agent5(Agent):
    def __init__(self, _interactions):
        """ Initialize our agent """
        super().__init__(_interactions)
        # These values give a nice demo with osoyoo car
        self._composite_interactions: list[CompositeInteraction] = []
        self._previous_interaction = None

    def action(self, _outcome):
        """Implement the agent's policy"""
        # tracing the previous cycle
        self._previous_interaction = self._last_interaction
        self._last_interaction = self._interactions[f"{self._intended_interaction.action}{_outcome}"]
        print(f"Action: {self._intended_interaction.action}, Prediction: {self._intended_interaction.outcome}, "
              f"Outcome: {_outcome}, Prediction_correct: {self._intended_interaction.outcome == _outcome}, "
              f"Valence: {self._last_interaction.valence})")

        # Recording previous composite interaction
        if self._previous_interaction is not None:
            composite_interaction = CompositeInteraction(self._previous_interaction, self._last_interaction)
            if composite_interaction not in self._composite_interactions:
                self._composite_interactions.append(composite_interaction)
                print(f"Learning {composite_interaction}")
            else:
                i = self._composite_interactions.index(composite_interaction)
                self._composite_interactions[i].reinforce()
                print(f"Reinforcing {self._composite_interactions[i]}")

        # Selecting the next action to enact
        proclivity_list = [0, 0, 0]
        # The composite interactions whose pre_interaction matches the last_interaction
        activated_interactions = \
            [ci for ci in self._composite_interactions if ci.pre_interaction == self._last_interaction]
        # The proclivity for each action
        for ai in activated_interactions:
            proclivity_list[ai.post_interaction.action] += ai.weight * ai.post_interaction.get_valence()

        print("Proclivity list: ", proclivity_list)
        # Select the action that has the max proclivity
        max_proclivity = max(proclivity_list)
        action = proclivity_list.index(max_proclivity)

        # Find the intended interaction
        self._intended_interaction = self._interactions[f"{action}0"]
        # It is the post interaction of the activated composite interaction that has the highest weight
        weight = 0
        for composite_interaction in activated_interactions:
            if composite_interaction.weight > weight and composite_interaction.post_interaction.action == action:
                weight = composite_interaction.weight
                self._intended_interaction = composite_interaction.post_interaction

        activated_keys = \
            [ci.key() for ci in self._composite_interactions if ci.pre_interaction == self._last_interaction]
        df = pd.DataFrame(activated_interactions, columns=['composite'])


        return action
