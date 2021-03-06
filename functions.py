import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from numpy import array
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def one_hot_preprocess(df):
    # sort values to be always same
    df = df.sort_values('SalePrice')

    # chech which are qualitative
    qualitative = [f for f in df.columns if df.dtypes[f] == 'object']

    # for every qualitative
    for name in qualitative:
        # get set of values and change nan with 'NA'
        distinct_values = df[name].unique()
        distinct_values = ['NA' if x is np.nan else x for x in distinct_values]
        distinct_values = array(distinct_values)

        # convert values to integer
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(distinct_values)

        # convert to one hot
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        # take whole column
        column = df[name].fillna('NA')
        column = column.map(lambda x: map_to_one_hot(x, onehot_encoded, distinct_values))

        # change current df column with one hot
        df[name] = column

    # all data sorted and one-hotted
    return df


def map_to_one_hot(row_value, onehot_encoded, distinct_values):
    row_value_index = np.where(distinct_values == row_value)[0][0]
    return onehot_encoded[row_value_index]


def encode(frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0] + 1)
    ordering = ordering['ordering'].to_dict()

    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature + '_E'] = o


def write_outliers_to_file(df):
    f = open('outliers.txt', 'w')
    for attrib in df:
        f.write(attrib + '\t\t\t\t\t')
        for index, row in df.iterrows():
            if row['GrLivArea'] > 4000:
                f.write(str(row[attrib]) + '\t')
        f.write('\n')
    f.close()


def clear_missing_data(train):
    # zbieranie info o brakujacych danych
    total = train.isnull().sum().sort_values(ascending=False)
    total = total[total > 0]
    percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
    percent = percent[percent > 0]
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    # print(missing_data)
    missing_plot = total.plot.bar()
    missing_plot.set_xlabel('Nazwa atrybutu')
    missing_plot.set_ylabel('Ilość rekordów')
    missing_plot.set_title('Brakujące dane')
    plt.show()

    # usuniecie rekordow z brakujacymi danymi
    ret_train = train.drop((missing_data[missing_data['Percent'] > 0.5]).index, 1)
    ret_train = ret_train.drop(train.loc[train['Electrical'].isnull()].index)
    return ret_train


# korelacja wszystkich atrybutow
def correlation_all(train):
    corrmat = train.corr()
    plt.subplots(figsize=(12, 9))
    sb.heatmap(corrmat, vmax=.8, square=True)
    plt.show()


# korelacja SalePrice z k-1 innymi atrybutami
def correlation_sale_price(train, k):
    corrmat = train.corr()
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(train[cols].values.T)
    sb.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
               xticklabels=cols.values)
    plt.show()


# Jednowymiarowo, metoda kwartylowa
def univ_outlier(data_set, atrib_name):
    dane_atrybut = data_set[atrib_name]
    mean = np.mean(dane_atrybut)
    q = np.percentile(dane_atrybut, 75) - np.percentile(dane_atrybut, 25)
    y1 = 2. * q + mean
    y2 = -1. * 2 * q + mean

    fig, ax = plt.subplots()
    ax.scatter(range(len(dane_atrybut)), dane_atrybut, label="Elementy zmiennej")
    ax.plot((0, len(dane_atrybut)), (y1, y1), 'r:', label='Granica Dolna Metody Kwartylowej')
    ax.plot((0, len(dane_atrybut)), (y2, y2), 'r--', label='Granica Górna Metody Kwartylowej')
    ax.set_xlabel('Numer elementu')
    ax.set_ylabel('Wartość elementu')
    ax.set_title('Elementy odosobnione zmiennej ' + atrib_name)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


# Dwuwymiarowo, metoda odległości
# Wybieramy p elementów których k-ty najbliższy sąsiad ma największą wartość
def biv_outlier(data_set, atrib_name_1, atrib_name_2, p, k):
    x = data_set[atrib_name_1]
    y = data_set[atrib_name_2]
    xy = list(zip(x, y))

    scaler = StandardScaler()
    x_n = scaler.fit_transform(np.reshape(np.array(x), (-1, 1)))
    x_n = np.squeeze(np.reshape(x_n, (1, -1)))
    y_n = scaler.fit_transform(np.reshape(np.array(y), (-1, 1)))
    y_n = np.squeeze(np.reshape(y_n, (1, -1)))
    xy_n = list(zip(x_n, y_n))

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xy_n)
    neighbor_matrix = nbrs.kneighbors_graph(xy_n, mode='distance')

    max_neighbor_indices = list(enumerate(np.squeeze(neighbor_matrix.argmax(1).tolist())))
    max_neighbor_distance = [neighbor_matrix[i] for i in max_neighbor_indices]
    argsort_max_neighbor_distance = np.argsort(max_neighbor_distance)[-p:]

    max_neighbor_indices = np.array(max_neighbor_indices)[argsort_max_neighbor_distance]

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Zależność " + atrib_name_2 + " od " + atrib_name_1)

    xy_outliers = np.array(xy)[max_neighbor_indices[:, 0]]
    ax.plot(xy_outliers[:, 0], xy_outliers[:, 1], 'ro', fillstyle='none', markersize=20, label="Elementy Odosobnione")
    ax.set_xlabel(atrib_name_1)
    ax.set_ylabel(atrib_name_2)
    ax.set_title('Elementy odosobnione zmiennych ' + atrib_name_1 + ' oraz ' + atrib_name_2)
    # ax.legend()
    plt.show()


def bad_cluster(train, quantitative, qualitative):
    # model = TSNE(n_components=2, random_state=0, perplexity=50)
    x = train[quantitative].fillna(0.).values
    onehot = train[qualitative].values
    # tsne = model.fit_transform(x)

    std = StandardScaler()
    s = std.fit_transform(x)
    s = np.concatenate([s, onehot], axis=1)
    pca = PCA(n_components=2)
    pca.fit(s)
    pc = pca.transform(s)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pc)

    # fr = pd.DataFrame({'tsne1': tsne[:, 0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})
    # sb.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)
    # plt.show()
    # print(np.sum(pca.explained_variance_ratio_))


def good_cluster(train):
    features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF',
                'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
    model = TSNE(n_components=2, random_state=0, perplexity=50)
    x = train[features].fillna(0.).values
    tsne = model.fit_transform(x)

    std = StandardScaler()
    s = std.fit_transform(x)
    pc = s
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(pc)

    fr = pd.DataFrame({'tsne1': tsne[:, 0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})
    sb.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)
    plt.show()


# klasteryzacja od Marzęci
# wersja select_dtypes i z wybranymi features bardzo podobne (hint?)
def cluster(train, k):
    # features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF',
    #             'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
    # s = train[features].fillna(0.).values
    s = train.fillna(0.).values

    scaler = MinMaxScaler()
    s_scaled = scaler.fit_transform(s)

    pca = PCA(n_components=2).fit(s_scaled)
    pca_2d = pca.transform(s_scaled)
    pca_2d_x = [p[0] for p in pca_2d]
    pca_2d_y = [p[1] for p in pca_2d]

    # pca = PCA(n_components=30).fit(s_scaled)
    # pca_test = pca.transform(s_scaled)
    print(s_scaled.shape)
    kmeans = KMeans(n_clusters=k).fit(s_scaled)

    # K = range(1,15)
    # sum_of_sq_dist = []
    # for k in K:
    #     km = KMeans(n_clusters=k)
    #     km = km.fit(s_scaled)
    #     sum_of_sq_dist.append(km.inertia_)
    #
    # plt.plot(K, sum_of_sq_dist, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Sum_of_squared_distances')
    # plt.title('Elbow Method For Optimal k')
    # plt.show()

    s_scaled_centers = kmeans.cluster_centers_
    s_scaled_centers_indexes = []
    for scaled_center in s_scaled_centers:
        euclid_distances = []
        for ind, scaled in enumerate(s_scaled):
            euclid_distances.append(euclidean(scaled, scaled_center))
        s_scaled_centers_indexes.append(np.argmin(euclid_distances))

    clusters_centers = pca.transform(kmeans.cluster_centers_)
    clusters_centers_x = [p[0] for p in clusters_centers]
    clusters_centers_y = [p[1] for p in clusters_centers]

    labels = kmeans.labels_
    plt.figure(figsize=(15, 15))
    plt.scatter(pca_2d_x, pca_2d_y, c=labels)
    plt.scatter(clusters_centers_x, clusters_centers_y, c='red', linewidths=10)
    plt.show()

    return s_scaled_centers_indexes


# klasyfikacja
# plan na klasyfikcaje
# 0. sortujemy wartości w treningowym zbiorze
# - kasujemy kolumny nie policzalne (tekstowe, etc)
# - jako klasy budujemy przedziały na podstawie ceny domu (przedziały do ustalenia)
# - zbior testowy z klasami zbudowanymi jedziemy KNearestClassifierem
# - sprawdzamy jak to wygląda na zbiorze treningowym
# - jeśli dobrze to jeszcze sprawdzamy dowolny przypadek podany przez nas
def classification(df_train, df_test):
    train_sorted = df_train.sort_values(by='SalePrice')

    # usun wartosci nie numeryczne w treningowym
    are_cols_numbers = df_train.dtypes != 'object'
    are_cols_numbers = are_cols_numbers.values
    str_columns_numbers = df_train.columns[are_cols_numbers]
    train_sorted = train_sorted[str_columns_numbers]

    # usun wartosci nie numeryczne w testowym
    are_cols_numbers = df_test.dtypes != 'object'
    are_cols_numbers = are_cols_numbers.values
    str_columns_numbers = df_test.columns[are_cols_numbers]
    df_test = df_test[str_columns_numbers]

    sale_price = train_sorted['SalePrice'].values
    train_sorted = train_sorted.drop(columns='SalePrice')

    # przedziały dla każdej wartości z SalePrice
    dep_count = 10
    dep_delta = abs(sale_price.min() - sale_price.max()) / dep_count
    min_dep_thresh = list(enumerate(range(int(sale_price.min()), int(sale_price.max()), int(dep_delta))))

    # konwersja SalePrice do kategorii
    sale_price_classes = []
    for price in sale_price:
        for ind, minimum in reversed(min_dep_thresh):
            if price >= minimum:
                sale_price_classes.insert(0, ind)
                break

    train_sorted = train_sorted.fillna(0)
    df_test = df_test.fillna(0)

    train_sorted_train, train_sorted_test, sale_price_classes_train, sale_price_classes_test = train_test_split(
        train_sorted, sale_price_classes)

    classifier = KNeighborsClassifier()
    classifier.fit(train_sorted_train, sale_price_classes_train)

    print('Klasyfikacja')
    print('Jakość klasyfikacji na zbiorze treningowym: ',
          classifier.score(train_sorted_train, sale_price_classes_train))
    print('Jakość klasyfikacji na zbiorze testowym: ', classifier.score(train_sorted_test, sale_price_classes_test))

    # predykcja... jaki przedział cenowy dla domów o danych parametrach
    classifier = KNeighborsClassifier()
    classifier.fit(train_sorted, sale_price_classes)
    predictions = classifier.predict(df_test)
    df_test = df_test.join(pd.DataFrame({'Klasa przedziału cenowego': predictions}))
    print('\nPredykcja cen nieruchomości')
    print('Przedziały cenowe, oraz odpowiadająca im minimalna cena', min_dep_thresh)
    df_test.to_csv('prediction.csv', encoding='utf-8')


def propose_price(variable_names, probe_record, orig_dataframe):
    # weź tylko wybrane dane
    df = orig_dataframe[variable_names]

    for name in variable_names:
        print(df[name].describe())

    pd.set_option('display.max_columns', None)
    print(df.loc[df['SalePrice'] == 625000])

    # oddziel SalePrice bo to bedzie klasa
    df_class = df['SalePrice']
    df = df.drop(columns='SalePrice')

    # zamien df_class na onehoty też
    df = pd.get_dummies(df)
    dummy_cols = df.columns

    # klasteryzacja zbioru otrzymanego
    cluster(df, 4)

    # przeskaluj
    scaler = MinMaxScaler()
    scaler.fit(df)
    df_scaled = scaler.transform(df)

    # cv_scores = []
    # for k in range(1, 11):
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     scores = cross_val_score(knn, df_scaled, df_class, cv=10, scoring='accuracy')
    #     cv_scores.append(scores.mean())
    # print(cv_scores)
    # # changing to misclassification error
    # MSE = [1 - x for x in cv_scores]
    #
    # # determining best k
    # optimal_k = MSE.index(min(MSE))
    # print("The optimal number of neighbors is %d" % optimal_k)
    # zbuduj klasyfikator
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(df_scaled, df_class)

    # zbuduj wiersz z danymi
    variable_names.remove('SalePrice')
    dict = {}
    for i in range(len(dummy_cols)):
        if dummy_cols[i] in variable_names:
            dict[dummy_cols[i]] = probe_record[variable_names.index(dummy_cols[i])]
        elif str.split(dummy_cols[i], '_')[1] in probe_record:
            dict[dummy_cols[i]] = 1
        else:
            dict[dummy_cols[i]] = 0

    to_test = pd.DataFrame(dict, index=[0])
    to_test_scaled = scaler.transform(to_test)

    output = classifier.predict(to_test_scaled)
    print(output)
