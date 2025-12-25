from typing import List
import math

def compute_budget(total_budget: int, citizen_votes: List[List[int]]) -> List[int]:
    """
    Computes budget using the General Median Algorithm with binary search.
    
    >>> compute_budget(100, [[100, 0, 0], [0, 0, 100]])
    [50, 0, 50]
    >>> compute_budget(100, [[80, 20], [20, 80]])
    [50, 50]
    >>> sum(compute_budget(100, [[33, 33, 34], [10, 10, 80]]))
    100
    >>> no_binary_search_compute_budget(30, [[0, 0, 6, 0, 0, 6, 6, 6, 6], [0, 6, 0, 6, 6, 6, 6, 0, 0], [6, 0, 0, 6, 6, 0, 0, 6, 6]])
    [0, 0, 0, 5, 5, 5, 5, 5, 5]
    """
    if not citizen_votes:
        return []

    num_subjects = len(citizen_votes[0])
    votes_per_subject = [sorted([citizen[i] for citizen in citizen_votes]) for i in range(num_subjects)]
    
    # Breakpoints where the slope of the total budget function can change
    breakpoints = {0}
    for votes in votes_per_subject:
        breakpoints.update(votes)
    sorted_breaks = sorted(list(breakpoints))

    def get_subject_allocation(votes: List[int], x: float) -> float:
        # Median of {0, v1, ..., vn, x}
        # In a list of N+2 elements, the median is the average of the two middle elements
        # if N+2 is even, or the middle element if odd.
        arr = sorted(votes + [0, x])
        n = len(arr)
        if n % 2 == 1:
            return float(arr[n // 2])
        else:
            return (arr[n // 2 - 1] + arr[n // 2]) / 2.0

    def get_total_and_slope(x: float):
        total = 0.0
        slope = 0
        eps = 1e-8
        for votes in votes_per_subject:
            v_now = get_subject_allocation(votes, x)
            v_next = get_subject_allocation(votes, x + eps)
            total += v_now
            # If the value increased, the slope for this subject is active (0.5 or 1.0)
            if v_next > v_now:
                slope += (v_next - v_now) / eps
        return total, slope

    # Find the linear segment [x_low, x_high] containing total_budget
    target_x = 0.0
    for i in range(len(sorted_breaks) - 1):
        x_low, x_high = sorted_breaks[i], sorted_breaks[i+1]
        sum_low, slope_low = get_total_and_slope(x_low)
        sum_high, _ = get_total_and_slope(x_high)
        
        if sum_low <= total_budget <= sum_high + 1e-9:
            if slope_low > 1e-9:
                target_x = x_low + (total_budget - sum_low) / slope_low
            else:
                target_x = x_low
            break
    else:
        # Extrapolate beyond the largest vote
        sum_last, slope_last = get_total_and_slope(sorted_breaks[-1])
        if slope_last > 1e-9:
            target_x = sorted_breaks[-1] + (total_budget - sum_last) / slope_last
        else:
            target_x = sorted_breaks[-1]

    # Calculate final floats and use Largest Remainder Method for integers
    floats = [get_subject_allocation(v, target_x) for v in votes_per_subject]
    ints = [int(math.floor(f + 1e-11)) for f in floats]
    
    remainder = total_budget - sum(ints)
    if remainder > 0:
        # Tie-breaking: fractional part descending, then index ascending
        indices = sorted(range(num_subjects), key=lambda i: (floats[i] - ints[i], -i), reverse=True)
        for i in range(int(round(remainder))):
            ints[indices[i]] += 1
            
    return ints


def no_binary_search_compute_budget(total_budget: int, citizen_votes: List[List[int]]) -> List[int]:
    """
    Computes budget using the General Median Algorithm by solving piecewise linear intervals.
    
    >>> no_binary_search_compute_budget(100, [[100, 0, 0], [0, 0, 100]])
    [50, 0, 50]
    >>> no_binary_search_compute_budget(100, [[80, 20], [20, 80]])
    [50, 50]
    >>> sum(no_binary_search_compute_budget(100, [[10, 20, 70], [30, 30, 40]]))
    100
    >>> no_binary_search_compute_budget(30, [[0, 0, 6, 0, 0, 6, 6, 6, 6], [0, 6, 0, 6, 6, 6, 6, 0, 0], [6, 0, 0, 6, 6, 0, 0, 6, 6]])
    [0, 0, 0, 5, 5, 5, 5, 5, 5]
    """
    if not citizen_votes:
        return []

    num_subjects = len(citizen_votes[0])
    votes_per_subject = [sorted([citizen[i] for citizen in citizen_votes]) for i in range(num_subjects)]
    
    # Breakpoints are where the slope changes
    breakpoints = {0}
    for votes in votes_per_subject:
        breakpoints.update(votes)
    sorted_breaks = sorted(list(breakpoints))

    def get_subject_allocation(votes: List[int], x: float) -> float:
        # Median of {0, v1, ..., vn, x}
        arr = sorted(votes + [0, x])
        n = len(arr)
        if n % 2 == 1:
            return float(arr[n // 2])
        else:
            return (arr[n // 2 - 1] + arr[n // 2]) / 2.0

    def get_total_and_slope(x: float):
        total = 0.0
        slope = 0.0
        eps = 1e-8
        for votes in votes_per_subject:
            v_now = get_subject_allocation(votes, x)
            v_next = get_subject_allocation(votes, x + eps)
            total += v_now
            # Calculate the local slope (rate of change)
            slope += (v_next - v_now) / eps
        return total, slope

    target_x = 0.0
    for i in range(len(sorted_breaks) - 1):
        x_low, x_high = sorted_breaks[i], sorted_breaks[i+1]
        sum_low, slope_low = get_total_and_slope(x_low)
        sum_high, _ = get_total_and_slope(x_high)
        
        # Check if the total budget falls within this linear segment
        if sum_low <= total_budget <= sum_high + 1e-9:
            if slope_low > 1e-9:
                target_x = x_low + (total_budget - sum_low) / slope_low
            else:
                target_x = x_low
            break
    else:
        # Handle cases where budget exceeds the sum at the highest vote
        sum_last, slope_last = get_total_and_slope(sorted_breaks[-1])
        if slope_last > 1e-9:
            target_x = sorted_breaks[-1] + (total_budget - sum_last) / slope_last
        else:
            target_x = sorted_breaks[-1]

    # Final allocation and integer rounding using Largest Remainder Method
    floats = [get_subject_allocation(v, target_x) for v in votes_per_subject]
    ints = [int(math.floor(f + 1e-11)) for f in floats]
    
    remainder = total_budget - sum(ints)
    if remainder > 0:
        # Sort by fractional part (descending) and use index for stability
        indices = sorted(range(num_subjects), key=lambda i: (round(floats[i] - ints[i], 10), -i), reverse=True)
        for i in range(int(round(remainder))):
            ints[indices[i % num_subjects]] += 1
            
    return ints


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)