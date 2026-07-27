"""Small helpers shared by the public examples."""

import mastermlx


def check_release():
    """Print the installed library version used by the example."""

    print(f"Using mastermlx {mastermlx.__version__}")
