import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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


def clear_missing_data(train):
    # zbieranie info o brakujacych danych
    total = train.isnull().sum().sort_values(ascending=False)
    total = total[total > 0]
    percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
    percent = percent[percent > 0]
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    # print(missing_data)
    total.plot.bar()
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
    ax.legend()
    plt.show()


# realizacja odległościowej metody szukania elementów odosobnionych
# wybieramy p elementów których k-ty najbliższy sąsiad ma największą wartość
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
    ax.legend()
    plt.show()


def bad_cluster(train, quantitative, qual_encoded):
    features = quantitative + qual_encoded
    features.remove('LotFrontage')
    features.remove('MasVnrArea')
    features.remove('GarageYrBlt')
    model = TSNE(n_components=2, random_state=0, perplexity=50)
    x = train[features].fillna(0.).values
    tsne = model.fit_transform(x)

    std = StandardScaler()
    s = std.fit_transform(x)
    pca = PCA(n_components=30)
    pca.fit(s)
    pc = pca.transform(s)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pc)

    fr = pd.DataFrame({'tsne1': tsne[:, 0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})
    sb.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)
    plt.show()
    # print(np.sum(pca.explained_variance_ratio_))


def good_cluster(train):
    features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF',
                'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
    # print(features)
    # print(len(features))
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


# klasyfikacja
# plan na klasyfikcaje
# 0. sortujemy wartości w treningowym zbiorze
# - kasujemy kolumny nie policzalne (tekstowe, etc)
# - jako klasy budujemy przedziały na podstawie ceny domu (przedziały do ustalenia)
# - zbior testowy z klasami zbudowanymi jedziemy KNearestClassifierem
# - sprawdzamy jak to wygląda na zbiorze treningowym
# - jeśli dobrze to jeszcze sprawdzamy dowolny przypadek podany przez nas.
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
