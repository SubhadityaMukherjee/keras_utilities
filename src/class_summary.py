from collections import Counter

from matplotlib import pyplot as plt


def class_summary(y):
    """
    Plot the class distribution of the data
    """
    w = Counter(y)
    plt.bar(w.keys(), w.values())
    # Set x-axis label
    plt.xlabel("Class")
    # Set y-axis label
    plt.ylabel("Number of data points")
    # Set title
    plt.title("Original Data Distribution")
    plt.show()
    # Set x values to class names
    # plt.xticks(positions,values)
