import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


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


def biv_outlier(data_set, atrib_name_1, atrib_name_2):
    x = data_set[atrib_name_1]
    y = data_set[atrib_name_2]
    xy = list(zip(x, y))
    print(xy)
    nbrs = NearestNeighbors(n_neighbors=3).fit(xy)
    neighbor_matrix = nbrs.kneighbors_graph(xy, mode='distance')
    print(neighbor_matrix)

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Zależność " + atrib_name_2 + " od " + atrib_name_1)
    ax.set_xlabel(atrib_name_1)
    ax.set_ylabel(atrib_name_2)
    ax.legend()
    plt.show()
