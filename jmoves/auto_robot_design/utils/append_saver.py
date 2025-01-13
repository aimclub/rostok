import os
import numpy as np


def chunk_list(lst, chunk_size):
    """Yield successive chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def save_result_append(filename, new_data):
    """
    Save new data to a .npz file, appending it to existing data if the file already exists.
    For each save, complete data is loaded.
    Parameters:
    filename (str): The name of the .npz file to save the data to. The filename must end with '.npz'.
    new_data (dict): A dictionary where keys are the names of the arrays and values are the numpy arrays to be saved.

    Raises:
    Exception: If the filename does not end with '.npz'.

    Example:
    >>> new_data = {'array1': np.array([[1, 2, 3]]), 'array2': np.array([[4, 5, 6]])}
    >>> save_result_append('data.npz', new_data)
    
    If 'data.npz' already exists and contains arrays with the same keys as in `new_data`, the new arrays are stacked
    vertically with the existing arrays. If 'data.npz' does not exist, it is created with the new data.

    """
    filename_str = str(filename)
    if not filename_str.endswith(".npz"):
        raise Exception("Must end with .npz")

    if os.path.exists(filename):
        # Load existing data
        existing_data = np.load(filename)
        # Append new data
        combined_data = {key: np.row_stack(
            (existing_data[key], new_data[key])) for key in new_data.keys()}
    else:
        combined_data = new_data
    # Save combined data back to file
    np.savez(filename, **combined_data)
