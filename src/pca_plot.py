from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns


def plot_pca(ds, labels):
    """Plots the first two and three principal components of the dataset.

    Args:
        ds (Dataframe): Number of features in the dataset. Needs to be of the form (n_samples, n_features)
        labels (Dataframe): Number of labels in the dataset. Needs to be of the form (n_samples, 1)
    """

    pca = PCA(n_components=3)
    # Normalize the dataset so that each feature is in the range [0, 1]
    # print('Normalizing the dataset...')
    # ds = StandardScaler().fit_transform(ds)
    print("Transforming the dataset...")
    pca_features = pca.fit_transform(ds)
    pcas = pd.DataFrame(pca_features, columns=["PC1", "PC2", "PC3"])

    # Add the first three principal components as a new column into a new dataframe
    pcas["PC1"] = pca_features[:, 0]
    pcas["PC2"] = pca_features[:, 1]
    pcas["PC3"] = pca_features[:, 2]

    # Add the labels to the dataframe
    pcas["label"] = labels

    print(
        "Explained variation per principal component: {}".format(
            pca.explained_variance_ratio_
        )
    )

    rndperm = np.random.permutation(pcas.shape[0])

    # Make two dimensional scatter plot
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="label",
        palette=sns.color_palette("hls", 5),
        data=pcas.loc[rndperm, :],
        legend="full",
        alpha=0.3,
    )
    plt.show()

    # Make three dimensional scatter plot
    ax = plt.subplot(111, projection="3d")
    ax.scatter(
        pcas.iloc[rndperm, 0],
        pcas.iloc[rndperm, 1],
        pcas.iloc[rndperm, 2],
        c=pcas.iloc[rndperm, 3],
        cmap=plt.cm.get_cmap("jet", 10),
    )
    ax.set_title("PCA of SIFT descriptors")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.show()

    plt.figure(figsize=(16, 10))
    # Make two dimensional scatter plot
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="label",
        # Set color palette
        palette=sns.color_palette("hls", 10),
        data=pcas.loc[rndperm, :],
        legend="full",
        alpha=0.3,
    )
    plt.show()
