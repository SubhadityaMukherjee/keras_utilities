def vocabulary_size(y_train):
    """
    Find maximum length and the size of the vocabulary in the training data.
    """
    train_labels_cleaned = []
    characters = set()
    max_len = 0

    for label in y_train:
        for char in label:
            characters.add(char)

        max_len = max(max_len, len(label))
        train_labels_cleaned.append(label)

    print("Maximum length: ", max_len)
    print("Vocab size: ", len(characters))
    return train_labels_cleaned, max_len, characters
