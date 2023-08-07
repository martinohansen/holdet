def levenshtein_distance(s1, s2):
    # Implement the Levenshtein distance algorithm here
    # See https://en.wikipedia.org/wiki/Levenshtein_distance for more information
    m = len(s1)
    n = len(s2)
    d = [[0 for j in range(n + 1)] for i in range(m + 1)]
    for i in range(1, m + 1):
        d[i][0] = i
    for j in range(1, n + 1):
        d[0][j] = j
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[m][n]


def get_closest_match(name, choices):
    # Find the player name with the smallest Levenshtein distance
    closest_match = None
    min_distance = float("inf")
    for choice in choices:
        distance = levenshtein_distance(name, choice)
        if distance < min_distance:
            closest_match = choice
            min_distance = distance
    return closest_match
