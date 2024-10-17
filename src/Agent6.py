import pandas as pd
from agent5_base import CompositeInteraction, Agent


class Agent6:
    def __init__(self, _interactions):
        """ Initialize our agent """
        self._interactions = {interaction.key(): interaction for interaction in _interactions}
        self._composite_interactions = {}
        self._intended_interaction = self._interactions["00"]
        self._last_interaction = None
        self._previous_interaction = None
        pd.set_option('display.max_columns', 8)
        default_interactions = [interaction for interaction in _interactions if interaction.outcome == 0]
        data = {'post_action': [i.action for i in default_interactions],
                'weight': [0] * len(default_interactions),
                'proclivity': [0] * len(default_interactions),
                'post_interaction': [i.key() for i in default_interactions]}
        self.default_df = pd.DataFrame(data)

    def action(self, _outcome):
        """Implement the agent's policy"""
        # tracing the previous cycle
        self._previous_interaction = self._last_interaction
        self._last_interaction = self._interactions[f"{self._intended_interaction.action}{_outcome}"]
        print(f"Action: {self._intended_interaction.action}, Prediction: {self._intended_interaction.outcome}, "
              f"Outcome: {_outcome}, Prediction_correct: {self._intended_interaction.outcome == _outcome}, "
              f"Valence: {self._last_interaction.valence}")

        # Recording previous composite interaction
        if self._previous_interaction is not None:
            composite_interaction = CompositeInteraction(self._previous_interaction, self._last_interaction)
            if composite_interaction.key() not in self._composite_interactions:
                self._composite_interactions[composite_interaction.key()] = composite_interaction
                print(f"Learning {composite_interaction}")
            else:
                self._composite_interactions[composite_interaction.key()].reinforce()
                print(f"Reinforcing {self._composite_interactions[composite_interaction.key()]}")

        # # Selecting the next action to enact
        # proclivity_list = [0, 0, 0]
        # # The composite interactions whose pre_interaction matches the last_interaction
        # activated_interactions = \
        #     [ci for ci in self._composite_interactions.values() if ci.pre_interaction == self._last_interaction]
        # # The proclivity for each action
        # for ai in activated_interactions:
        #     proclivity_list[ai.post_interaction.action] += ai.weight * ai.post_interaction.get_valence()
        #
        # # print("Proclivity list: ", proclivity_list)
        # # Select the action that has the max proclivity
        # max_proclivity = max(proclivity_list)
        # action = proclivity_list.index(max_proclivity)
        #
        # # Find the intended interaction
        # self._intended_interaction = self._interactions[f"{action}0"]
        # # It is the post interaction of the activated composite interaction that has the highest weight
        # weight = 0
        # for composite_interaction in activated_interactions:
        #     if composite_interaction.weight > weight and composite_interaction.post_interaction.action == action:
        #         weight = composite_interaction.weight
        #         self._intended_interaction = composite_interaction.post_interaction

        activated_keys = \
            [ci.key() for ci in self._composite_interactions.values() if ci.pre_interaction == self._last_interaction]

        data = {'composite': activated_keys,
                'weight': [self._composite_interactions[k].weight for k in activated_keys],
                'post_valence': [self._composite_interactions[k].post_interaction.valence for k in activated_keys],
                'post_action': [self._composite_interactions[k].post_interaction.action for k in activated_keys],
                'post_interaction': [self._composite_interactions[k].post_interaction.key() for k in activated_keys]
                }
        df = pd.DataFrame(data)

        # df = pd.DataFrame(activated_keys, columns=['composite'])
        # df['weight'] = df['composite'].apply(lambda x: self._composite_interactions[x].weight)
        # df['post_valence'] = df['composite'].apply(lambda x: self._composite_interactions[x].post_interaction.valence)
        # df['post_action'] = df['composite'].apply(lambda x: self._composite_interactions[x].post_interaction.action)
        # df['post_interaction'] = df['composite'].apply(lambda x: self._composite_interactions[x].post_interaction.key())

        # Add the actions by default
        df = pd.concat([self.default_df, df], ignore_index=True)

        # Compute the proclivity for each action
        df['proclivity'] = df['weight'] * df['post_valence']
        grouped_df = df.groupby('post_action').agg({'proclivity': 'sum'}).reset_index()
        df = df.merge(grouped_df, on='post_action', suffixes=('', '_sum'))

        # Find the most probable outcome for each action
        max_weight_df = df.loc[df.groupby('post_action')['weight'].idxmax(), ['post_action', 'post_interaction']].reset_index(drop=True)
        max_weight_df.columns = ['post_action', 'intended']
        df = df.merge(max_weight_df, on='post_action')

        df = df.drop(['proclivity', 'post_interaction'], axis=1)
        df = df.rename(columns={'proclivity_sum': 'proclivity'})
        # print(df)
        print(df[['composite', 'weight', 'post_valence', 'post_action', 'proclivity', 'intended']])

        # Find the first row that has the highest proclivity
        max_index = df['proclivity'].idxmax()
        intended_interaction = df.loc[max_index, ['intended']].values[0]
        self._intended_interaction = self._interactions[intended_interaction]
        print("Intended", self._intended_interaction)
        return self._intended_interaction.action
        # return action
