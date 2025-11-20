import math
from typing import List
from scipy.optimize import linear_sum_assignment


def envy_free_room_allocation(valuations: List[List[float]], rent: float, verbose: bool = False):
    """
    Compute an envy-free allocation and prices for room-rent division.

    Doctests
    --------

    >>> eps = 1e-9
    >>> vals = [[150,0],[140,10]]
    >>> assignment, prices = envy_free_room_allocation(vals, rent=100)
    >>> assignment == {0: 0, 1: 1}
    True
    >>> # Check envy-freeness
    >>> all(vals[i][assignment[i]] - prices[i] >= vals[i][assignment[j]] - prices[j] - eps
    ...     for i in range(2) for j in range(2))
    True

    >>> vals = [[10,10],[10,10]]
    >>> assignment, prices = envy_free_room_allocation(vals, rent=50)
    >>> sorted(assignment.values())
    [0, 1]
    >>> # Everyone is indifferent → still envy-free
    >>> all(vals[i][assignment[i]] - prices[i] >= vals[i][assignment[j]] - prices[j] - eps
    ...     for i in range(2) for j in range(2))
    True

    >>> vals = [[100, 40, 20],
    ...         [ 90, 30, 10],
    ...         [ 70, 60, 50]]
    >>> assignment, prices = envy_free_room_allocation(vals, rent=120)
    >>> sorted(assignment.values())
    [0, 1, 2]
    >>> all(vals[i][assignment[i]] - prices[i] >= vals[i][assignment[j]] - prices[j] - eps
    ...     for i in range(3) for j in range(3))
    True

    >>> # zero rent
    >>> vals = [[5,1],[6,2]]
    >>> assignment, prices = envy_free_room_allocation(vals, rent=0)
    >>> sum(prices) == 0
    True

    >>> all(vals[i][assignment[i]] - prices[i] >= vals[i][assignment[j]] - prices[j] - eps
    ...     for i in range(2) for j in range(2))
    True

    >>> # possible negative payments, still envy-free
    >>> vals = [[100,0],[20,10]]
    >>> assignment, prices = envy_free_room_allocation(vals, rent=30)
    >>> all(vals[i][assignment[i]] - prices[i] >= vals[i][assignment[j]] - prices[j] - eps
    ...     for i in range(2) for j in range(2))
    True

    """

    n = len(valuations)

    # -----------------------------
    # STEP 1 — Welfare-maximizing assignment via Hungarian algorithm
    # -----------------------------
    # Hungarian solves a *minimization* problem, so negate valuations.
    cost = [[-v for v in row] for row in valuations]
    players, rooms = linear_sum_assignment(cost)

    # assignment[i] = room assigned to player i
    assignment = {p: r for p, r in zip(players, rooms)}

    # -----------------------------
    # STEP 2 — Compute envy-free prices using potentials via Bellman–Ford
    # -----------------------------
    # Build directed graph on players:
    # edge i → j has weight:
    #    w(i→j) = v(i, room_i) – v(i, room_j)
    #
    # The "potential" for player i is the max path weight leaving i.

    # Build edge list
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ri = assignment[i]
            rj = assignment[j]
            w = valuations[i][ri] - valuations[i][rj]
            edges.append((i, j, w))

    # Bellman–Ford: compute longest-path potentials
    # (negate weights since BF normally computes MIN distance)
    def longest_paths_from(src):
        dist = [-math.inf] * n
        dist[src] = 0

        # Relax edges n−1 times
        for _ in range(n - 1):
            improved = False
            for u, v, w in edges:
                if dist[u] != -math.inf and dist[u] + w > dist[v]:
                    dist[v] = dist[u] + w
                    improved = True
            if not improved:
                break
        return dist

    potentials = [max(longest_paths_from(i)) for i in range(n)]

    # Convert potentials into preliminary prices (they are envy-free)
    # Payment_i = potential_i
    prices = potentials[:]

    # -----------------------------
    # Adjust prices so they sum to the rent
    # -----------------------------
    total_price = sum(prices)
    shift = (total_price - rent) / n
    prices = [p - shift for p in prices]

    # -----------------------------
    # Print result
    # -----------------------------
    if verbose:
        for i in range(n):
            r = assignment[i]
            print(
                f"Player {i} gets room {r} with value {valuations[i][r]}, "
                f"and pays {prices[i]:.2f}"
            )
        eps = 1e-9
        print("\nVerifying envy-freeness:")
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                vi = valuations[i][assignment[i]] - prices[i]
                vj = valuations[i][assignment[j]] - prices[j]
                print(
                    f"Player {i} values their own room at {vi:.2f} "
                    f"and room of player {j} at {vj:.2f}"
                )
                
                assert vi >= vj - eps, f"Envy detected: Player {i} envies Player {j} because {vi} < {vj}"

    return assignment, prices

if __name__ == "__main__":
    import doctest

    doctest.testmod()