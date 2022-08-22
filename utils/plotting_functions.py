import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

from plotly.offline import iplot
import matplotlib
import numpy as np
import seaborn as sns


def plot_normal_distribution(dist_list: list, label_list: list = None):
    plt.style.use('seaborn')

    if label_list is None:
        label_list = list(range(len(dist_list)))
    numpy_list = list()
    for i in dist_list:
        linspace = np.linspace(i[0] - 3 * i[1], i[0] + 3 * i[1], 10000)
        dist = stats.norm.pdf(linspace, i[0], i[1])
        numpy_list.append(dist)
        plt.plot(linspace,dist, label=str(i[0]))
    data = np.stack(numpy_list).transpose()
    df = pd.DataFrame(data, columns=label_list)

    #plt.plot(df)
    plt.legend()
    #plt.xlim(df.min().min(),df.max().max())
    plt.show()
    # iplot(df)
    # plot = sns.displot(data, kind='kde', rug=True).fig.show()
    return None


if __name__ == "__main__":
    test = [(0, 0.5), (1, 0.2), (0.5, 1.5)]
    test2 = plot_normal_distribution(test)
