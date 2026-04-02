import numpy as np
import pandas as pd

def main():
    data = pd.read_csv("datasets/dataset_example.csv")
    print(data.columns)
    """
    Index(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday',
       'Best Hand', 'Arithmancy', 'Astronomy', 'Herbology',
       'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
       'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
       'Care of Magical Creatures', 'Charms', 'Flying'],
      dtype='str')
    """

    # Drop the 'Index' column and convert the rest to a NumPy array
    data = data.drop(columns=['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'])
    # print(data.columns)

    features = data.drop(columns=['Hogwarts House'])
    houses = data['Hogwarts House']

    print(type(features), features.shape, features, features.loc[0])  # <class 'numpy.ndarray'>
    print(type(houses), houses.shape, houses)  # <class 'numpy.ndarray'> (n_samples,)

    features_ndarray = features.to_numpy()
    houses_ndarray = houses.to_numpy()

    print("------------------")
    print("ndarray:")
    print(f'Type: {type(features_ndarray)}, Shape: {features_ndarray.shape}') # (n_samples, n_features)
    print(f'{features_ndarray}')
    print(f'{features_ndarray[0]}') # Access the first data point (first row)
    print(f'{features_ndarray[0][0]}') # Access the first feature of the first data point (first row, first column)
    
    for data_point in features_ndarray:
        print(f'Type: {type(data_point)}, {data_point}') # Type: <class 'numpy.ndarray'>, [5.0, 3.0, 1.0, 0.2]

    print("------------------")
    print("Test adding 2 features:")
    added_features = features_ndarray[0] + features_ndarray[1]
    print(f'{features_ndarray[0]} + {features_ndarray[1]} = {added_features}')

if __name__ == "__main__":
    main()