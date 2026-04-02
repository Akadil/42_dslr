import numpy as np

def main():
    # Create a 2D numpy array (matrix) with shape (3, 2)
    matrix = np.random.randint(0, 4, size=(10, 1))
    labels = np.array([1, 2, 3, 4])


    print(matrix)

    matrix_magix = matrix[:, None] # shape (3, 1, 2)
    print(f'Matrix with expanded dimensions: {matrix_magix.shape}')
    # print(matrix_magix)

    y_binary = (matrix[:, None] == labels)
    print(f'Binary labels: {y_binary.shape}')
    print(f'Labels: {labels}')
    print(y_binary)

if __name__ == "__main__":
    main()