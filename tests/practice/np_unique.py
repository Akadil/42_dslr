import numpy as np

"""
np.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)

Finds the unique elements of an array and optionally returns:
- The unique sorted elements
- The indices of ar that result in the unique array
- The indices of the unique array that can be used to reconstruct ar
- The number of times each unique element appears in ar

Parameters:
    ar: Input array (will be flattened if axis is None)
    return_index (bool): If True, return indices of ar that result in the unique array
    return_inverse (bool): If True, return indices to reconstruct ar from unique array
    return_counts (bool): If True, return count of each unique element
    axis (int or None): Axis along which to find unique elements

Returns:
    unique (ndarray): The sorted unique elements of the array
    unique_indices (ndarray): Optional - indices of first occurrence of unique values
    unique_inverse (ndarray): Optional - indices to reconstruct original array
    unique_counts (ndarray): Optional - count of occurrences for each unique element

Example:
    np.unique([1, 2, 1, 3, 2]) -> array([1, 2, 3])
    np.unique([1, 2, 1, 3, 2], return_counts=True) -> (array([1, 2, 3]), array([2, 2, 1]))
"""
# Example 1: Basic usage - find unique elements
arr = np.array([[1, 2, 1, 3, 2, 1], [1, 2, 4, 3, 2, 1]])
unique_elements = np.unique(arr)
print("Unique elements:", unique_elements, type(unique_elements))

# Example 2: Get indices of first occurrences
unique_elements, indices = np.unique(arr, return_index=True)
print("Unique elements:", unique_elements)
print("First occurrence indices:", indices)

# Example 3: Get indices to reconstruct original array
unique_elements, inverse_indices = np.unique(arr, return_inverse=True)
print("Unique elements:", unique_elements)
print("Inverse indices:", inverse_indices)
print("Reconstructed array:", unique_elements[inverse_indices])

# Example 4: Get counts of each unique element
unique_elements, counts = np.unique(arr, return_counts=True)
print("Unique elements:", unique_elements)
print("Counts:", counts)

# Example 5: All returns at once
unique_elements, indices, inverse_indices, counts = np.unique(
    arr, return_index=True, return_inverse=True, return_counts=True
)
print("All results:", unique_elements, indices, inverse_indices, counts)

# Example 6: 2D array with axis
arr_2d = np.array([[1, 2], [1, 2], [3, 4]])
unique_rows = np.unique(arr_2d, axis=0)
print("Unique rows:\n", unique_rows)