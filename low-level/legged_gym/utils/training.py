def remaining_learning_iterations(target_iteration, current_iteration):
    """Return the number of iterations needed to reach an absolute target."""
    return max(0, int(target_iteration) - int(current_iteration))
