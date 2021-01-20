from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# X: [number of images x number of clusters]
# y: label for each image

def findClassifier(classifier_name, X, y):

    if classifier_name == 'SVM':

        model = SVC()

        parameter_space = {
            'C': [100, 200, 500, 1000, 2000],
            'kernel': ['linear'],
            'class_weight': ['balanced'],
        }

        svm = GridSearchCV(model, parameter_space, cv=5)
        svm.fit(X, y)

        return svm

    if classifier_name == 'MLP':

        model = MLPClassifier()

        parameter_space = {
            'hidden_layer_sizes': [(10, 30, 10), (100,)],
            'max_iter': [300, 400, 500],
            'learning_rate': ['constant', 'adaptive'],
        }

        mlp = GridSearchCV(model, parameter_space, cv=5)
        mlp.fit(X, y)

        return mlp

    if classifier_name == 'KNN':
        model = 'To be added'
        
def TrainClassifier(classifier_name, parameters, X, y):

    if classifier_name == 'SVM':
        
        # To find best parameters for SVM
        # model = findClassifier(classifier_name)
        # return model
        
        c = parameters[0]
        model = SVC(C=c, kernel='linear', class_weight='balanced')
        return model.fit(X,y)

    elif classifier_name == 'MLP':
        
        # To find best parameters for MLP
        # model = findClassifier(classifier_name, X, y)
        # return model

        hls = parameters[0]
        model = MLPClassifier(hidden_layer_sizes=hls, learning_rate='adaptive', max_iter=300,)
        return model.fit(X, y)
        
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
