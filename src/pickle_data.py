import pickle


def save_to_pickle(data, filename):
    # save the data to a file
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        print('saved dataset to file: "{}"'.format(f.name))


def load_from_pickle(filename):
    # load the data from a file
    with open(filename, "rb") as f:
        return pickle.load(f)
