import pickle
from typing import Any


def pickle_load(path: str) -> Any:
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except pickle.UnpicklingError:
        raise pickle.UnpicklingError(f"Error loading the pickle file: {path}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading {path}: {e}")
