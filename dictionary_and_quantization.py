import numpy as np
from scipy.cluster.vq import vq
from numpy.linalg import norm
from sklearn.cluster import KMeans


def findDictionary(data, k=100, max_iter=10, method='libKmeans'):

    if method=='impKmeans':
        # random initialization
        centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
        ind = np.zeros((data.shape[0], 1))
    
        for iter_num in range(max_iter):
    
            for i in range(data.shape[0]):
                delta = norm((data[i, :] - centroids), axis=1)
                ind[i, :] = np.argmin(delta)
    
            for j in range(k):
                centroids[j, :] = np.mean(data[ind[:, 0] == j, :], axis=0)
            print("K-Means Iteration Number:", iter_num)
    elif method=='libKmeans':
        centroids = KMeans(n_clusters=k, n_init=2, random_state=0, max_iter=400).fit(data)
        return centroids


def featureQuantization(descriptor_list, centroids, k=100, method='libKmeans'):
    if method=='impKmeans':
        num_images = len(descriptor_list)
        num_words = centroids.shape[0]
        
        im_features = np.zeros((num_images, num_words), "float32")
        for i in range(num_images):
            words, _ = vq(descriptor_list[i], centroids)
            for w in words:
                im_features[i][w] += 1
                
        # Normalize the image features such that L1 norm of each image is 1:
        im_features = im_features / np.sum(np.abs(im_features), axis=1).reshape((im_features.shape[0], 1))
    elif method=='libKmeans':
        features = []
        for i in range(len(descriptor_list)):
            clust_assigns = centroids.predict(descriptor_list[i])
            feature = np.histogram(clust_assigns, np.arange(k+1))
            features.append(list(feature[0]))

        im_features = np.array(features)
        im_features = im_features / np.sum(np.abs(im_features) , axis=1).reshape((im_features.shape[0],1))
    
    return im_features