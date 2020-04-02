import numpy as np
import pandas as pd
from winning_prob import prob_reach_set_score


def test_set_score_probs_add_to_one():

    rally_prob_a = np.random.uniform(0, 1)
    rally_prob_b = np.random.uniform(0, 1)

    outcome_dict = dict()

    for poss_outcomes in [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 5),
                          (7, 6)]:

        outcome_dict[poss_outcomes] = prob_reach_set_score(
            *poss_outcomes, rally_prob_a, rally_prob_b)
        outcome_dict[tuple(reversed(poss_outcomes))] = prob_reach_set_score(
            *reversed(poss_outcomes), rally_prob_a, rally_prob_b)

    summed = pd.Series(outcome_dict).sum()

    assert np.allclose(1., summed)
