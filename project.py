import pandas as pd

from functions import *

df_train = pd.read_csv('train.csv')

# metoda kwartylowa, zmienna general living area
# atrybut = 'GrLivArea'
# univ_outlier(df_train, atrybut)

# metoda odległości, zmienna general living area vs sales price
atryb1, atryb2 = 'GrLivArea', 'SalePrice'
biv_outlier(df_train, atryb1, atryb2)
