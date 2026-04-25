def _sort_key(x):
    """Sort key: numeric values first (ascending), then alphabetic strings."""
    try:
        return (0, float(x))
    except (ValueError, TypeError):
        return (1, str(x))
