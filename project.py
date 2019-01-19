from functions import *

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train_cluster = pd.read_csv('train.csv')

quantitative = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in df_train.columns if df_train.dtypes[f] == 'object']

qual_encoded = []
for q in qualitative:
    encode(df_train_cluster, q)


''' 1 obsluga brakujacych parametrow '''
df_train = clear_missing_data(df_train)

''' 2 korelacja '''
correlation_all(df_train)
correlation_sale_price(df_train, 10)

'''3 univ i biv elementy odosobnione'''
# metoda kwartylowa, zmienna general living area
atrybut = 'GrLivArea'
univ_outlier(df_train, atrybut)

# metoda odległości, zmienna general living area vs sale price
atryb1, atryb2 = 'GrLivArea', 'SalePrice'
biv_outlier(df_train, atryb1, atryb2, 10, 3)

''' 4a zla klasteryzacja '''
bad_cluster(df_train_cluster, quantitative, qual_encoded)

''' 4b dobra klasteryzacja '''
good_cluster(df_train_cluster)

''' 5 klasyfikacja '''
classification(df_train, df_test)
