from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import cv2 as cv
import numpy as np
from quantization import get_hist

# X: [number of images x number of clusters]
# y: label for each image

class KNN:
    def __init__(self, k):
        self.k = k
    def fit(self, X, y):
        self.num_cluster = X.shape[1]
        self.X = np.copy(X)
        self.y = np.copy(y)
        return self
    def predict(self, X_test):
        y_predict = np.zeros((X_test.shape[0],))
        for i, x in enumerate(X_test):
            # distance between the histogram and all cluster
            norms = np.linalg.norm(self.X - x, axis=1)
            # minimum k indices
            idx = np.argpartition(norms, self.k)[:self.k]
            # minimum k distances
            dists = norms[idx]
            # corresponding labels
            labels = self.y[idx]
            values, counts = np.unique(labels, return_counts=True)
            # maximum occuring labels
            occurs = values[counts == np.max(counts)]
            if len(occurs) == 1:
                y_predict[i] = occurs[0]
            else:
                # multiple labels are candidate for being the closest k
                # choose the one with minimum distance
                mask = np.isin(labels, occurs)
                y_predict[i] = labels[mask][np.argmin(dists[mask])]
        return y_predict

def TrainClassifier(classifier_name, parameters, X, y):

    if classifier_name == 'SVM':
        c = parameters[0]
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        model = SVC(C=c, kernel='linear', class_weight='balanced')
        return model.fit(X,y)

    elif classifier_name == 'MLP':
        layers = parameters[0]
        lr = parameters[1]
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        model = MLPClassifier(hidden_layer_sizes = layers, learning_rate = lr)
        return model.fit(X,y)

    elif classifier_name == 'KNN':
        k = parameters[0]
        model = KNN(k)
        # to use the external library
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        # from sklearn.neighbors import KNeighborsClassifier
        # model = KNeighborsClassifier(n_neighbors=k)
        return model.fit(X, y)

    else:
        raise Exception('Invalid option')

def TestClassifier(classifier_name, model, X_test):

    if classifier_name == 'SVM':
        return model.predict(X_test)

    elif classifier_name == 'MLP':
        return model.predict(X_test)

    elif classifier_name == 'KNN':
        return model.predict(X_test)

    else:
        raise Exception('Invalid option')

def func_compute_histograms(impath, indices, descriptor, quantizer, X, y):
    taxon_id, img_id = indices
    img = cv.imread(impath)
    assert img is not None
    hist = get_hist(img, descriptor, quantizer)
    if hist is None:
        return
    X.append(hist)
    y.append(taxon_id)

if __name__ == '__main__':
    import imgops
    from timeit import default_timer as timer

    """from descriptors import get_descriptor
    from quantization import get_quantizer
    from dictionary import get_dictionary
    # parameters
    quantname = 'BOW'
    dictname = 'BOW'
    num_cluster = 100
    descname = 'SIFT'
    desc_per_img = 20
    # initialize objects
    quants = imgops.load_quants(quantname, dictname, num_cluster,
                                descname, desc_per_img)
    descriptor = get_descriptor(descname)
    quantizer = get_quantizer(quantname, descriptor)
    vocab = imgops.load_vocab(dictname, num_cluster, descname, desc_per_img)
    quantizer.setVocabulary(vocab)
    # obtain histogram of each train image
    start = timer()
    X = []
    y = []
    imgops.loop_images(func_compute_histograms, (descriptor, quantizer, X, y))
    X = np.vstack(X)
    end = timer()
    # print and save the results
    print('Computing histograms took %.1f seconds' % (end - start))
    print('X:', X.shape)
    imgops.save_array('X', X)
    imgops.save_array('y', y)"""

    X = np.load(imgops.NPYDIR + 'X.npy')
    y = np.load(imgops.NPYDIR + 'y.npy')
    X_test = X
    # classification parameters
    classifier_name, parameters = 'SVM', [1.0]
    classifier_name, parameters = 'MLP', [100, 'constant']
    classifier_name, parameters = 'KNN', [5]
    # train the classifier
    start = timer()
    model = TrainClassifier(classifier_name, parameters, X, y)
    end = timer()
    print('Train took %.1f seconds' % (end - start))
    # test the classifier
    start = timer()
    predict = TestClassifier(classifier_name, model, X_test)
    end = timer()
    # print results
    print('Self test took %.1f seconds' % (end - start))
    accuracy = np.count_nonzero(predict == y) / len(y)
    print('accuracy: %.1f%%' % (100 * accuracy))
