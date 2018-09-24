import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.optimize import minimize


def hold_serve_prob(rally_win_prob):

    rally_lose_prob = 1 - rally_win_prob

    term_1 = pow(rally_win_prob, 4)
    term_2 = (1+4*rally_lose_prob+10*(pow(rally_lose_prob, 2)))

    first_summand = term_1 * term_2

    term_1 = 20 * pow(rally_win_prob * rally_lose_prob, 3)
    term_2 = pow(rally_win_prob, 2)
    term_3 = 1/(1 - 2*rally_win_prob*rally_lose_prob)

    second_summand = term_1 * term_2 * term_3

    result = first_summand + second_summand

    return result


def prob_reach_tiebreak_score(i, j, win_serve_rally_prob_a,
                              win_serve_rally_prob_b):

    # Helpful renamings

    lose_serve_rally_prob_a = 1 - win_serve_rally_prob_a
    lose_serve_rally_prob_b = 1 - win_serve_rally_prob_b

    a_served_last = (((i - 1 + j) % 4 == 0) or ((i - 1 + j) % 4 == 3))

    # Initial conditions:

    if (i == 0 and j == 0):
        return 1

    if (i < 0 or j < 0):
        return 0

    if a_served_last:

        total = 0

        if (not (i == 7 and j <= 6)):

            total += prob_reach_tiebreak_score(
                i, j-1, win_serve_rally_prob_a, win_serve_rally_prob_b) * (
                    lose_serve_rally_prob_a)

        if (not (j == 7 and i <= 6)):

            total += (prob_reach_tiebreak_score(
                i-1, j, win_serve_rally_prob_a, win_serve_rally_prob_b) *
                win_serve_rally_prob_a)

        return total

    else:

        total = 0

        if (not (i == 7 and j <= 6)):

            total += (prob_reach_tiebreak_score(
                i, j-1, win_serve_rally_prob_a, win_serve_rally_prob_b) *
                win_serve_rally_prob_b)

        if (not (j == 7 and i <= 6)):

            total += (prob_reach_tiebreak_score(i-1, j, win_serve_rally_prob_a,
                                                win_serve_rally_prob_b) *
                      lose_serve_rally_prob_b)

        return total


def prob_win_tiebreak_a(win_serve_rally_prob_a, win_serve_rally_prob_b):

    total = 0
    lose_serve_rally_prob_a = 1 - win_serve_rally_prob_a
    lose_serve_rally_prob_b = 1 - win_serve_rally_prob_b

    for j in range(6):

        total += prob_reach_tiebreak_score(7, j, win_serve_rally_prob_a,
                                           win_serve_rally_prob_b)

    total += (prob_reach_tiebreak_score(6, 6, win_serve_rally_prob_a,
                                        win_serve_rally_prob_b) *
              win_serve_rally_prob_a * lose_serve_rally_prob_b *
              1/(1 - win_serve_rally_prob_a * win_serve_rally_prob_b -
                 lose_serve_rally_prob_a * lose_serve_rally_prob_b))

    return total


def prob_win_set_a(win_serve_rally_prob_a, win_serve_rally_prob_b):

    hold_serve_prob_a = hold_serve_prob(win_serve_rally_prob_a)
    hold_serve_prob_b = hold_serve_prob(win_serve_rally_prob_b)

    lose_serve_prob_b = 1 - hold_serve_prob_b

    total = 0

    for j in range(5):

        total += prob_reach_set_score(6, j, hold_serve_prob_a,
                                      hold_serve_prob_b)

    total += (prob_reach_set_score(6, 5, hold_serve_prob_a, hold_serve_prob_b)
              * lose_serve_prob_b)

    total += (prob_reach_set_score(6, 6, hold_serve_prob_a, hold_serve_prob_b)
              * prob_win_tiebreak_a(win_serve_rally_prob_a,
                                    win_serve_rally_prob_b))

    return total


def prob_reach_set_score(i, j, hold_serve_prob_a, hold_serve_prob_b):

    # Helpful renamings

    lose_serve_prob_a = 1 - hold_serve_prob_a
    lose_serve_prob_b = 1 - hold_serve_prob_b

    a_served_last = ((i - 1 + j) % 2 == 0)

    # Initial conditions

    if (i == 0 and j == 0):
        return 1

    if (i < 0 or j < 0):
        return 0

    # Two possibilities

    if a_served_last:

        total = 0

        if (not (j == 6 and i <= 5)):

            total += (prob_reach_set_score(i-1, j, hold_serve_prob_a,
                                           hold_serve_prob_b) *
                      hold_serve_prob_a)

        if (not (i == 6 and j <= 5)):

            total += (prob_reach_set_score(i, j-1, hold_serve_prob_a,
                                           hold_serve_prob_b) *
                      lose_serve_prob_a)

        return total

    else:

        total = 0

        if (not (j == 6 and i <= 5)):

            total += (prob_reach_set_score(i-1, j, hold_serve_prob_a,
                                           hold_serve_prob_b) *
                      lose_serve_prob_b)

        if (not (i == 6 and j <= 5)):

            total += (prob_reach_set_score(i, j-1, hold_serve_prob_a,
                                           hold_serve_prob_b) *
                      hold_serve_prob_b)

        return total


def prob_win_match_a(win_serve_rally_prob_a, win_serve_rally_prob_b,
                     best_of_five=False):

    prob_a_win_set = prob_win_set_a(win_serve_rally_prob_a,
                                    win_serve_rally_prob_b)

    prob_b_win_set = prob_win_set_a(win_serve_rally_prob_b,
                                    win_serve_rally_prob_a)

    total = 0

    if (not best_of_five):

        total += pow(prob_a_win_set, 2)
        total += 2 * pow(prob_a_win_set, 2) * prob_b_win_set

    else:

        total += (pow(prob_a_win_set, 3) + 3 * pow(prob_a_win_set, 3) *
                  prob_b_win_set + 6 * pow(prob_a_win_set, 3) *
                  pow(prob_b_win_set, 2))

    return total


def calculate_set_probabilities(p1_rally_prob, p1_rally_sigma, p2_rally_prob,
                                p2_rally_sigma, samples=5000):

    spw_1_samples = np.random.normal(p1_rally_prob, p1_rally_sigma, samples)
    spw_2_samples = np.random.normal(p2_rally_prob, p2_rally_sigma, samples)

    probabilities = np.zeros((samples, 7))
    prob_reach_tiebreak = np.zeros(samples)

    scores = [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 5), (7, 6)]

    for i, (spw_1, spw_2) in enumerate(zip(spw_1_samples, spw_2_samples)):

        hold_serve_prob_1 = hold_serve_prob(spw_1)
        hold_serve_prob_2 = hold_serve_prob(spw_2)

        prob_reach_tiebreak[i] = prob_reach_set_score(6, 6, hold_serve_prob_1,
                                                      hold_serve_prob_2)

        for no, (g1, g2) in enumerate(scores):

            if (g2 <= 4):
                probabilities[i, no] = prob_reach_set_score(
                    g1, g2, hold_serve_prob_1, hold_serve_prob_2)
            elif (g2 == 5):
                probabilities[i, no] = (prob_reach_set_score(6, 5,
                                                             hold_serve_prob_1,
                                                             hold_serve_prob_2)
                                        * (1 - hold_serve_prob_2))
            elif (g2 == 6):
                probabilities[i, no] = (prob_reach_set_score(6, 6,
                                                             hold_serve_prob_1,
                                                             hold_serve_prob_2)
                                        * prob_win_tiebreak_a(spw_1, spw_2))

    set_scores = np.average(probabilities, 0)
    prob_reach_tiebreak = np.average(prob_reach_tiebreak)

    prob_a_win_set = np.sum(set_scores)

    print("Prob of A winning set is: " + str(prob_a_win_set))

    straight_set_win = pow(prob_a_win_set, 3)
    four_set_win = 3 * pow(prob_a_win_set, 3) * (1 - prob_a_win_set)
    five_set_win = 6 * pow(prob_a_win_set, 3) * (1 - prob_a_win_set)**2
    win = straight_set_win + four_set_win + five_set_win

    print("Straight set win: " + str(straight_set_win))
    print("Four set win: " + str(four_set_win))
    print("Five set win: " + str(five_set_win))
    print("Overall: " + str(win))

    set_scores = {score: probability for score, probability in
                  zip(scores, set_scores)}

    return [set_scores, prob_reach_tiebreak, straight_set_win,
            four_set_win, five_set_win, win]


def solve_test():

    cons = ({'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 1})

    print(minimize(lambda x: abs(prob_win_match_a(0.59, x) - 0.6), [0.5],
                   constraints=cons))


if __name__ == '__main__':

    print(prob_reach_set_score(6, 4, 0.8, 0.8))
