import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def univ_outlier(data_set, atrib_name):
    dane_atrybut = data_set[atrib_name]
    mean = np.mean(dane_atrybut)
    Q = np.percentile(dane_atrybut, 75) - np.percentile(dane_atrybut, 25)
    y1 = 2. * Q + mean
    y2 = -1. * 2 * Q + mean

    fig, ax = plt.subplots()
    ax.scatter(range(len(dane_atrybut)), dane_atrybut, label="Elementy zmiennej")
    ax.plot((0, len(dane_atrybut)), (y1, y1), 'r:', label='Granica Dolna Metody Kwartylowej')
    ax.plot((0, len(dane_atrybut)), (y2, y2), 'r--', label='Granica Górna Metody Kwartylowej')
    ax.set_xlabel('Numer elementu')
    ax.set_ylabel('Wartość elementu')
    ax.set_title('Elementy odosobnione zmiennej ' + atrib_name)
    ax.legend()
    plt.show()


# realizacja odległościowej metody szukania elementów odosobnionych.
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
