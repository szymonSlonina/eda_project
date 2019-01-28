from functions import *

df_train_orig = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train_cluster = pd.read_csv('train.csv')

quantitative = [f for f in df_train_orig.columns if df_train_orig.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in df_train_orig.columns if df_train_orig.dtypes[f] == 'object']

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

# df_train_orig = df_train_orig.drop(columns='PoolQC')
# df_train_orig = df_train_orig.drop(columns='MiscFeature')
# df_train_orig = df_train_orig.drop(columns='Alley')
# df_train_orig = df_train_orig.drop(columns='Fence')
# df_train_orig = df_train_orig.drop(columns='Id')
#
# df_train = pd.get_dummies(df_train_orig)
# df_train.to_csv('train_onehot.csv')
''' 2 korelacja '''
# correlation_all(df_train)
# correlation_sale_price(df_train, 10)

''' 3 elementy odosobnione '''
# univ_outlier(df_train, atrib_name='GrLivArea')
# biv_outlier(df_train, atrib_name_1='GrLivArea', atrib_name_2='SalePrice', p=10, k=3)

''' 4 klasteryzacja '''
# bad_cluster(df_train, quantitative, qualitative, qual_encoded)  # not working
# good_cluster(df_train_cluster)
# sr_wier_ind = cluster(df_train, 3)
# df_train_orig.loc[sr_wier_ind].to_csv('asdf.csv')

''' 5 klasyfikacja '''
# LotArea - całkowity wymiar posesji
# Street - typ dojazdu do posesji
# Utilities - udogodnienia działki
# Neighborhood - dzielnica
# HouseStyle - ilość pięter
# OverallQual - jakość materiałów i wykończenia
# OverallCond - jakość domu
# BsmtQual - wysokość piwnicy
# BsmtCond - stan piwnicy ogólnie
considered_variable_names = ['LotArea', 'Street', 'Utilities', 'Neighborhood', 'HouseStyle', 'OverallQual',
                             'OverallCond', 'BsmtQual', 'BsmtCond', 'GrLivArea', 'FullBath', 'BedroomAbvGr',
                             'KitchenAbvGr',
                             'GarageCars', 'GarageCond', 'SalePrice']
sample_record = [210000, 'Pave', 'AllPub', 'NoRidge', '2Story', 10, 9, 'Ex', 'Ex', 5642, 3, 8, 3, 4, 'Ex']
propose_price(considered_variable_names, sample_record, df_train_orig)
# classification(df_train, df_test)
