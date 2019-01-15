import pandas as pd
import functions_temp

df_train = pd.read_csv('train.csv')
df_train_cluster = pd.read_csv('train.csv')

quantitative = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in df_train.columns if df_train.dtypes[f] == 'object']

qual_encoded = []
for q in qualitative:
    functions_temp.encode(df_train_cluster, q)


''' 1 obsluga brakujacych parametrow '''
# df_train = functions_temp.clear_missing_data(df_train, True)

''' 2 korelacja '''
# functions_temp.correlation_all(df_train)
# functions_temp.correlation_sales_price(df_train)

''' 4a zla klasteryzacja '''
# functions_temp.bad_cluster(df_train_cluster, quantitative, qual_encoded)
