import numpy as np
import pandas as pd

def main():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    print(data)
    print(np.mean(data, axis=0))
    print(np.mean(data, axis=1))
    # print(np.mean(data, axis=2)) # segfault

if __name__ == "__main__":
    main()