from matplotlib import pyplot as plt
import seaborn as sns


def plot_class_distr(x_train, Y_train):
    """
    Plot the class distribution of the data
    """
    plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor="w", edgecolor="r")
    sns.set(font_scale=2)
    sns.barplot(x_train, Y_train)
    plt.xticks(color="w")
    plt.title("Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Number of Instances")
    plt.show()
    plt.savefig("distribution.pdf")
