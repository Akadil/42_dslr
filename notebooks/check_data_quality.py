import pandas as pd

df = pd.read_csv("datasets/dataset_train.csv", index_col=0)

# Missing values per column
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)  #

print(pd.DataFrame({"missing": missing, "percentage": missing_pct})[missing > 0])

"""Results:
                               missing   percentage
Arithmancy                          34  2.12
Astronomy                           32  2.00
Herbology                           33  2.06
Defense Against the Dark Arts       31  1.94
Divination                          39  2.44
Muggle Studies                      35  2.19
Ancient Runes                       35  2.19
History of Magic                    43  2.69
Transfiguration                     34  2.12
Potions                             30  1.88
Care of Magical Creatures           40  2.50
"""
