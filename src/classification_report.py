import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def pred_data(model, X_test, y_test):
    y_pred = model.predict(X_test, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_bool))
    return accuracy_score(y_test, y_pred_bool)
