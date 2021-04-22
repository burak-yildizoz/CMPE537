from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
import numpy as np

def Scores(y_test, y_predict):
    precision, recall, F1, support = precision_recall_fscore_support(y_test, y_predict)
    MeanF1 = np.mean(F1)
    conf_matrix = multilabel_confusion_matrix(y_test, y_predict)

    return MeanF1, precision, recall, F1, conf_matrix
