import numpy as np


def main():
    data = np.array([1, 2, 3, 4, 5, 6])
    print("Original data:", data)

    chunks = np.split(data, 3)
    print("Chunks:", chunks)

    new_chunks = np.array_split(data, len(data) // 4 + 1)
    print("New Chunks:", new_chunks)

if __name__ == "__main__":
    main()