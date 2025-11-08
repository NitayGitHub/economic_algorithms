#!python3

import cvxpy

def find_print_eglatarian_division(mat):
    """
    >>> mat = [
    ...     [81, 19, 1],
    ...     [70, 1, 29]
    ... ]
    >>> find_print_eglatarian_division(mat)
    Agent #1 gets: 0.53 of resource #1, 1.00 of resource #2, 0.00 of resource #3.
    Agent #2 gets: 0.47 of resource #1, 0.00 of resource #2, 1.00 of resource #3.
    """
    m = len(mat[0]) # number of resources
    n = len(mat)    # number of people

    resources = []
    
    for j in range(m):
        resources.append(cvxpy.Variable(n))

    utilities = []
    for i in range(n):
        utility = 0
        for j in range(m):
            utility += mat[i][j] * resources[j][i]
        utilities.append(utility)

    constraints = []
    for j in range(m):
        constraints.append(0 <= resources[j])
        constraints.append(resources[j] <= 1)
        constraints.append(cvxpy.sum(resources[j]) == 1)

    min_utility = cvxpy.Variable()
    prob = cvxpy.Problem(
        cvxpy.Maximize(min_utility),
        constraints = constraints + [min_utility <= utilities[i] for i in range(n)])
    prob.solve()

    for i in range(n):
        print(f"Agent #{i+1} gets: ", end="")
        for j in range(m):
            print(f"{resources[j][i].value:.2f} of resource #{j+1}", end=", " if j < m-1 else ".\n")

if __name__ == "__main__":
    mat = [
        [81, 19, 1],
        [70, 1, 29]
    ]
    find_print_eglatarian_division(mat)