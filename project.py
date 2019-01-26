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
# df_train = clear_missing_data(df_train)
# qualitative.remove('PoolQC')
# qualitative.remove('MiscFeature')
# qualitative.remove('Alley')
# qualitative.remove('Fence')
# df_train = one_hot_preprocess(df_train)

''' 2 korelacja '''
# correlation_all(df_train)
# correlation_sale_price(df_train, 10)

''' 3 elementy odosobnione '''
# univ_outlier(df_train, atrib_name='GrLivArea')
# biv_outlier(df_train, atrib_name_1='GrLivArea', atrib_name_2='SalePrice', p=10, k=3)

''' 4 klasteryzacja '''
# bad_cluster(df_train, quantitative, qualitative, qual_encoded)  # not working
# good_cluster(df_train_cluster)
cluster(df_train_cluster, 3)

''' 5 klasyfikacja '''
# classification(df_train, df_test)
