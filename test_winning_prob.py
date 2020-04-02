import numpy as np
import pandas as pd
from winning_prob import prob_reach_set_score, prob_win_set_a

poss_outcomes = [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 5), (7, 6)]


def test_set_score_probs_add_to_one():

    rally_prob_a = np.random.uniform(0, 1)
    rally_prob_b = np.random.uniform(0, 1)

    outcome_dict = dict()

    for poss_outcome in poss_outcomes:

        outcome_dict[poss_outcome] = prob_reach_set_score(
            poss_outcome[0], poss_outcome[1], rally_prob_a, rally_prob_b)
        outcome_dict[tuple(reversed(poss_outcome))] = prob_reach_set_score(
            tuple(reversed(poss_outcome))[0],
            tuple(reversed(poss_outcome))[1],
            rally_prob_a, rally_prob_b)

    summed = pd.Series(outcome_dict).sum()

    assert np.allclose(1., summed)


def test_prob_win_set_a_consistent():

    rally_prob_a = np.random.uniform(0, 1)
    rally_prob_b = np.random.uniform(0, 1)

    outcome_dict = dict()

    for poss_outcome in poss_outcomes:

        outcome_dict[poss_outcome] = prob_reach_set_score(
            poss_outcome[0], poss_outcome[1], rally_prob_a, rally_prob_b)

    set_win_prob = prob_win_set_a(rally_prob_a, rally_prob_b)

    assert np.allclose(set_win_prob, pd.Series(outcome_dict).sum())
