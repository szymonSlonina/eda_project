import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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


def clear_missing_data(train, print_plot=False):
    # zbieranie info o brakujacych danych
    total = train.isnull().sum().sort_values(ascending=False)
    total = total[total > 0]
    percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
    percent = percent[percent > 0]
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    if print_plot:
        print(missing_data)
        total.plot.bar()
        plt.show()

    # usuniecie rekordow z brakujacymi danymi
    ret_train = train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
    ret_train = ret_train.drop(train.loc[train['Electrical'].isnull()].index)
    return ret_train


def correlation_all(train):
    # korelacja wszystkich atrybutow
    corrmat = train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sb.heatmap(corrmat, vmax=.8, square=True)
    plt.show()


def correlation_sales_price(train):
    # korelacja SalePrice z k-1 innymi atrybutami
    k = 10
    corrmat = train.corr()
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(train[cols].values.T)
    sb.set(font_scale=1.25)
    hm = sb.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                    xticklabels=cols.values)
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
