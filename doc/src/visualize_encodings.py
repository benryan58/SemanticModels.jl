import argparse

from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import sklearn.cluster as cluster
from umap import UMAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize latent dimensional '
                                                 'encodings using UMAP.')
    parser.add_argument('--threeD', action="store_true",
                        help='Produce a 3D visualization')
    parser.add_argument('-d', '--data', required=True,
                        help='The encodings data for visualizing.')
    # parser.add_argument('--maxlen', default=500, type=int,
    #                     help='Maximum code snippet length.')
    # parser.add_argument('--dimension', default=64, type=int,
    #                     help='Encoding dimension for representation.')
    # parser.add_argument('--use_gpu', action="store_true",
    #                     help='Should we use the GPU if available?')
    # parser.add_argument('-m', '--mode', default='train', options=['train','encode']
    #                     help='Mode for auto-encoder [train, encode].')

    args = parser.parse_args()

    encoded_reps = pd.read_csv(args.data)
    X_test = encoded_reps[funcs.top_folder=="base"]
    Y_test = funcs.top2[funcs.top_folder=="base"]
    code_test = funcs.code[funcs.top_folder=="base"]
    top_cats = list(Y_test.value_counts()[Y_test.value_counts()>=100].index)

    X_test = X_test[Y_test.apply(lambda x: x in top_cats)]
    code_test = code_test[Y_test.apply(lambda x: x in top_cats)]
    Y_test = Y_test[Y_test.apply(lambda x: x in top_cats)]

    if args.threeD:
        n_comp = 3
    else:
        n_comp = 2

    reducer = UMAP(random_state=42, 
                   metric='cosine', 
                   n_neighbors=30, 
                   n_components=n_comp)
    embedding = reducer.fit_transform(X_test)


    # Clustering

    sils = silhouette_samples(X_test, Y_test, metric='cosine')
    clusts = pd.concat([X_test.reset_index(drop=True), 
                        Y_test.reset_index(drop=True), 
                        pd.Series(sils)], axis=1, ignore_index=True)
    centroids = clusts.groupby(64).agg('mean').sort_values(65)

    # sils2 = silhouette_samples(embedding, Y_test, metric='cosine')
    # clusts2 = pd.concat([X_test.reset_index(drop=True), 
    #                      Y_test.reset_index(drop=True), 
    #                      pd.Series(sils2)], axis=1, ignore_index=True)
    # centroids2 = clusts2.groupby(64).agg('mean').sort_values(65)

    src = list(centroids.index)

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    NUM_COLORS = len(src)
    step = int(np.floor(len(sorted_names)/NUM_COLORS))
    my_cols = [sorted_names[i] 
               for i in range(0, len(sorted_names), step)]

    if args.threeD:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax2 = fig.add_subplot(111, projection="3d")
        
        for i, s in enumerate(src):
            ax2.scatter(embedding_3d[Y_test==s, 0], 
                        embedding_3d[Y_test==s, 1], 
                        embedding_3d[Y_test==s, 2], 
                        c=my_cols[i], 
                        linewidths=0.1,
                        edgecolors='k',
                        label=s)
        plt.setp(ax2, xticks=[], yticks=[]) 
        plt.title("Julia source code data embedded into two dimensions by UMAP", 
                  fontsize=18) 
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))
        plt.subplots_adjust(right=0.75)
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(12, 10))

        for i, s in enumerate(src):
            ax.scatter(embedding[Y_test==s, 0], 
                        embedding[Y_test==s, 1], 
                        c=my_cols[i], 
                        linewidths=0.1,
                        edgecolors='k',
                        label=s)
        plt.setp(ax, xticks=[], yticks=[]) 
        plt.title("Julia source code data embedded into two dimensions by UMAP", 
                  fontsize=18) 
        plt.legend(loc="upper left", bbox_to_anchor=(1,1))
        plt.subplots_adjust(right=0.75)
        plt.show()

