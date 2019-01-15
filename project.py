import pandas as pd
import seaborn as sb

from functions import *

df_train = pd.read_csv('train.csv')

''' 1 obsluga brakujacych parametrow '''
# zbieranie info o brakujacych danych
total = df_train.isnull().sum().sort_values(ascending=False)
total = total[total > 0]
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
percent = percent[percent > 0]
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# print(missing_data)
# total.plot.bar()
# plt.show()

# usuniecie rekordow z brakujacymi danymi
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

''' 2 korelacja '''
# korelacja wszystkich atrybutow
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sb.heatmap(corrmat, vmax=.8, square=True)
plt.show()
#
# # korelacja SalePrice z k-1 innymi atrybutami
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
hm = sb.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                xticklabels=cols.values)
plt.show()

'''4a zla klasteryzacja'''

'''3 univ i biv elementy odosobnione'''
# metoda kwartylowa, zmienna general living area
atrybut = 'GrLivArea'
univ_outlier(df_train, atrybut)

# metoda odległości, zmienna general living area vs sales price
atryb1, atryb2 = 'GrLivArea', 'SalePrice'
biv_outlier(df_train, atryb1, atryb2, 10, 3)
