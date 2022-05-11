import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from yellowbrick.text import TSNEVisualizer

def plot_tsne(ds, save=False, dir="images/"):
    """Generates a t-SNE plot of the dataset.


    Args:
        ds (DataFrame)): Dateframe of any size with resized images.
    """
    # Flaten the images
    images = np.array(ds["image"].tolist())
    images = images.reshape(images.shape[0], -1)

    # Apply t-SNE
    tsne_ds = TSNE(n_components=2, random_state=42,
                   init='random', n_jobs=-1).fit_transform(images)

    # Normalize the data
    tsne_ds = (tsne_ds - tsne_ds.min()) / (tsne_ds.max() - tsne_ds.min())

    # Plot the t-SNE
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(tsne_ds[:, 0], tsne_ds[:, 1], c=ds["label"])
    ax.legend(ds["label"].unique())
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE visualization of the data")

    # if save save the plot to pdf
    if save:
        # Check if the directory exists
        if not os.path.exists(dir):
            # Make the directory
            os.makedirs(dir)

        plt.savefig(dir + "t-SNE_plot.pdf")

    plt.show()

    def plot_tsne_other(ds, column="image", save=True, save_dir="images/", file_name="tsne_projection.pdf"):
        """Other t-SNE plot, with legend of classes.

        Args:
            ds (DataFrame): Total dataset.
            save (bool, optional): Indicates whether to save file. Defaults to True.
            save_dir (str, optional): Target directory. Defaults to "images/".
            file_name (str, optional): Target filename. Defaults to "tsne_projection.pdf".
        """
        # Flaten the images
        images = np.array(ds["image"].tolist())
        images = images.reshape(images.shape[0], -1)

        # Apply t-SNE
        tsne = TSNEVisualizer()
        tsne.fit(images, ds["class_name"],  show=False)
        tsne.set_title("t-SNE projection of images")

        tsne.finalize()
        if save:
            # Check if the directory exists
            if not os.path.exists(dir):
                # Make the directory
                os.makedirs(dir)
            tsne.poof(outpath=save_dir + file_name)