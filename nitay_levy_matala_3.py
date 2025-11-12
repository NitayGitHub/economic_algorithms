import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse import csr_matrix

def max_product_assignment(mat, forbid_zero=True, eps=1e-12):
    """
    Finds an assignment that (approximately) maximizes the product of utilities.
    Assumes mat is a 2D list/array of shape (n_people, n_items) and n_people == n_items.
    
    Parameters
    ----------
    mat : array-like (n x n)
        Utility matrix u[i][j].
    forbid_zero : bool
        If True, treats zero utilities as forbidden (assigning them is avoided
        unless absolutely necessary) by giving them a huge cost. If False, zeros
        are replaced with eps and the algorithm will prefer small products to
        avoid infinite costs.
    eps : float
        Small positive number used to replace zeros if forbid_zero is False.
    
    Returns
    -------
    assignment : list of (person_index, item_index)
    total_product : float
    total_logsum : float  (sum of logs for the chosen assignment)

    Examples
    --------
    >>> mat = [[1, 2], [3, 4]]
    >>> assignment, total_product, total_logsum = max_product_assignment(mat)
    >>> assignment
    [(0, 1), (1, 0)]
    >>> round(total_product, 5)
    6.0
    >>> round(total_logsum, 5)
    1.79176
    """
    U = np.array(mat, dtype=float)
    n, m = U.shape
    if n != m:
        raise ValueError("This implementation requires a square matrix (n people, n items).")

    # Handle non-positive entries:
    if np.any(U < 0):
        raise ValueError("Matrix contains negative utilities. This function expects non-negative utilities.")

    # Replace zeros according to forbid_zero flag
    if forbid_zero:
        tiny = eps
        U_safe = U.copy()
        zero_mask = (U_safe <= 0)
        U_safe[zero_mask] = tiny
        cost = -np.log(U_safe)
        cost[zero_mask] = 1e9
    else:
        U_safe = U.copy()
        U_safe[U_safe <= 0] = eps
        cost = -np.log(U_safe)

    # linear_sum_assignment minimizes the total cost
    row_ind, col_ind = linear_sum_assignment(cost)

    chosen_values = U[row_ind, col_ind]
    total_logsum = np.sum(np.log(np.maximum(chosen_values, eps)))
    total_product = float(np.prod(chosen_values))

    assignment = list(zip(row_ind.tolist(), col_ind.tolist()))
    return assignment, total_product, total_logsum


def find_print_etlatarian_division(mat):
    """
    Prints the utilitarian (sum-maximizing) division using the Hungarian algorithm.

    Examples
    --------
    >>> mat = [[1, 2], [3, 4]]
    >>> find_print_etlatarian_division(mat)
    Agent #1 gets resource #2 (utility: 2)
    Agent #2 gets resource #1 (utility: 3)
    Total utility: 5
    """
    cost_matrix = -np.array(mat)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    total_utility = 0
    for i, j in zip(row_ind, col_ind):
        print(f"Agent #{i+1} gets resource #{j+1} (utility: {mat[i][j]})")
        total_utility += mat[i][j]
    
    print(f"Total utility: {total_utility}")


def egalitarian_assignment(mat, tol=1e-6):
    """
    Find an assignment that maximizes the minimum utility (egalitarian allocation).

    mat: n x n matrix of utilities (higher is better)
    Returns:
        best_threshold: the maximum guaranteed minimum utility
        assignment: list of (person, item)

    Examples
    --------
    >>> mat = [[1, 2], [3, 4]]
    >>> best_threshold, assignment = egalitarian_assignment(mat)
    >>> round(best_threshold, 5)
    2.0
    >>> sorted(assignment)
    [(0, 1), (1, 0)]
    """
    mat = np.array(mat, dtype=float)
    n = mat.shape[0]

    unique_values = np.unique(mat)
    lo, hi = 0, len(unique_values) - 1
    best_threshold = unique_values[0]
    best_assignment = None

    while lo <= hi:
        mid = (lo + hi) // 2
        threshold = unique_values[mid]

        adjacency = (mat >= threshold).astype(int)
        graph = csr_matrix(adjacency)

        match = maximum_bipartite_matching(graph, perm_type='column')
        if np.all(match != -1):  # perfect matching found
            best_threshold = threshold
            best_assignment = list(enumerate(match.tolist()))
            lo = mid + 1
        else:
            hi = mid - 1

    return best_threshold, best_assignment

if __name__ == "__main__":
    import doctest
    doctest.testmod()