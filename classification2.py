from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# X: [number of images x number of clusters]
# y: label for each image

def TrainClassifier(classifier_name, parameters, X, y):

    if classifier_name == 'SVM':
        c = parameters[0]
        model = SVC(C=c, kernel='linear', class_weight='balanced')
        return model.fit(X,y)

    elif classifier_name == 'MLP':
        layers = parameters[0]
        lr = parameters[1]
        max_iter = parameters[2]
        model = MLPClassifier(hidden_layer_sizes = layers, learning_rate = lr, max_iter=max_iter)
        return model.fit(X,y)

    elif classifier_name == 'kNN':
        k = parameters[0]
        return ('To be added')

    else:
        raise Exception('Invalid option')

def TestClassifier(classifier_name, model, X_test):

    if classifier_name == 'SVM':
        return model.predict(X_test)

    elif classifier_name == 'MLP':
        return model.predict(X_test)

    elif classifier_name == 'kNN':
        model.predict()
        return ('To be added')

    else:
        raise Exception('Invalid option')
