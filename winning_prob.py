

def hold_serve_prob(rally_win_prob):
    """Calculates the probability of holding serve.

    Args:
        rally_win_prob (np.array): The probability of winning a rally on serve.

    Returns:
        np.array: The probability of holding serve.
    """

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
    """The probability of reaching a given tiebreak score when player a serves
    first.

    Args:
        i (int): Score for player a.
        j (int): Score for player b.
        win_serve_rally_prob_a (np.array): The probability that a wins a rally
            on their serve.
        win_serve_rally_prob_b (np.array): The probability that b wins a rally
            on their serve.

    Returns:
        np.array: The probability of reaching tiebreak score [i, j].
    """

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
    """Calculates the probability that a wins a tiebreak.

    Args:
        win_serve_rally_prob_a (np.array): The probability that player a wins
            a rally on their own serve.
        win_serve_rally_prob_b (np.array): The probability that player b wins
            a rally on their own serve.

    Returns:
        np.array: The probability a wins the tiebreak.
    """

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
    """Calculates the probability that player a wins a set.

    Args:
        win_serve_rally_prob_a (np.array): The probability that player a wins
            a rally on their own serve.
        win_serve_rally_prob_b (np.array): The probability that player b wins
            a rally on their own serve.

    Returns:
        np.array: The probability that player a wins a set.
    """

    total = 0

    for j in range(5):

        total += prob_reach_set_score(6, j, win_serve_rally_prob_a,
                                      win_serve_rally_prob_b)

    total += prob_reach_set_score(7, 5, win_serve_rally_prob_a,
                                  win_serve_rally_prob_b)

    total += prob_reach_set_score(7, 6, win_serve_rally_prob_a,
                                  win_serve_rally_prob_b)

    return total


def prob_reach_set_score(i, j, win_serve_rally_prob_a, win_serve_rally_prob_b):
    """The probability of reaching a given set score when player A serves
    first.

    Args:
        i (int): Score for player a.
        j (int): Score for player b.
        win_serve_rally_prob_a (np.array): The probability that a wins a rally
            on their serve.
        win_serve_rally_prob_b (np.array): The probability that b wins a rally
            on their serve.

    Returns:
        np.array: The probability of reaching set score [i, j].
    """

    assert (j <= 6 and i <= 6) or (i == 7 and j <= 6) or (i <= 6 and j == 7), \
        'Please provide a valid set score!'

    hold_serve_prob_a = hold_serve_prob(win_serve_rally_prob_a)
    hold_serve_prob_b = hold_serve_prob(win_serve_rally_prob_b)

    # Helpful renamings
    lose_serve_prob_a = 1 - hold_serve_prob_a
    lose_serve_prob_b = 1 - hold_serve_prob_b

    a_served_last = ((i - 1 + j) % 2 == 0)

    # Initial conditions
    if (i == 0 and j == 0):
        return 1

    if (i < 0 or j < 0):
        return 0

    # We have the tiebreak case:
    if i == 6 and j == 7:
        return prob_reach_set_score(
            i, j - 1, win_serve_rally_prob_a, win_serve_rally_prob_b) * (
                1 - prob_win_tiebreak_a(win_serve_rally_prob_a,
                                        win_serve_rally_prob_b))

    elif i == 7 and j == 6:
        return prob_reach_set_score(
            i - 1, j, win_serve_rally_prob_a, win_serve_rally_prob_b) * (
                prob_win_tiebreak_a(win_serve_rally_prob_a,
                                    win_serve_rally_prob_b))

    # We also have the 7-5 case:
    if i == 7 and j == 5:
        return prob_reach_set_score(i - 1, j, win_serve_rally_prob_a,
                                    win_serve_rally_prob_b) * lose_serve_prob_b
    elif i == 5 and j == 7:
        return prob_reach_set_score(i, j - 1, win_serve_rally_prob_a,
                                    win_serve_rally_prob_b) * hold_serve_prob_b

    # Two possibilities
    if a_served_last:

        total = 0

        if (not (j == 6 and i <= 5)):

            total += (prob_reach_set_score(i-1, j, win_serve_rally_prob_a,
                                           win_serve_rally_prob_b) *
                      hold_serve_prob_a)

        if (not (i == 6 and j <= 5)):

            total += (prob_reach_set_score(i, j-1, win_serve_rally_prob_a,
                                           win_serve_rally_prob_b) *
                      lose_serve_prob_a)

        return total

    else:

        total = 0

        if (not (j == 6 and i <= 5)):

            total += (prob_reach_set_score(i-1, j, win_serve_rally_prob_a,
                                           win_serve_rally_prob_b) *
                      lose_serve_prob_b)

        if (not (i == 6 and j <= 5)):

            total += (prob_reach_set_score(i, j-1, win_serve_rally_prob_a,
                                           win_serve_rally_prob_b) *
                      hold_serve_prob_b)

        return total


def prob_win_match_a(win_serve_rally_prob_a, win_serve_rally_prob_b,
                     best_of_five=False):
    """Calculates the probability that player a wins the match.

    Args:
        win_serve_rally_prob_a (np.array): The probability that player a wins
            a rally on their own serve.
        win_serve_rally_prob_b (np.array): The probability that player b wins
            a rally on their own serve.
        best_of_five (Bool): Whether or not the match is in best-of-five
            format. If False, it is assumed to be best-of-three.

    Returns:
        np.array: The probability that player a wins the match.
    """

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


if __name__ == '__main__':

    print(prob_reach_set_score(6, 4, 0.8, 0.8))
