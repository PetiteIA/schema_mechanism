from agent4_base import Interaction


class Agent4:
    def __init__(self, _hedonist_table):
        """ Creating our agent """
        self.hedonist_table = [[2, -1], [1, -1], [1, -1]]
        self._action = 0
        self.anticipated_outcome = None

        self.memory = {}
        self.previous_interaction = None
        self.last_interaction = None

    def action(self, outcome):
        """ tracing the previous cycle """
        if self._action is not None:
            print("Action: " + str(self._action) +
                  ", Anticipation: " + str(self.anticipated_outcome) +
                  ", Outcome: " + str(outcome) +
                  ", Satisfaction: (anticipation: " + str(self.anticipated_outcome == outcome) +
                  ", valence: " + str(self.hedonist_table[self._action][outcome]) + ")")

        """ Recording previous experience """
        self.previous_interaction = self.last_interaction
        valence = self.hedonist_table[self._action][outcome]
        self.last_interaction = Interaction.create_or_retrieve(self._action, outcome, valence)
        # print("Enacted interaction ", end="")
        # print(self.last_interaction)
        if self.previous_interaction is not None:
            self.memory[self.previous_interaction] = self.last_interaction

        """ Computing the next action to enact """
        self._action = 0
        self.anticipated_outcome = None

        if self.last_interaction in self.memory:
            proposed_interaction = self.memory[self.last_interaction]
            if proposed_interaction is not None:
                if proposed_interaction.valence > 0:
                    self._action = proposed_interaction._action
                    self.anticipated_outcome = proposed_interaction.outcome
                else:
                    if proposed_interaction._action == 0:
                        self._action = 1
                    #elif proposed_interaction.action == 1:
                    #    self._action = 2
                    else:
                        self._action = 0
                    self.anticipated_outcome = None

        return self._action
