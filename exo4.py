"""
TP 2 - Deep Learning avec Keras et Manifold Untangling
Exercice 4 : Visualisation avec t-SNE
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.datasets import mnist
from divers import *

# Création de l'ensemble d'apprentissage et de test
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Calcul T-SNE
X_embedded = TSNE(n_components=2, perplexity=30.0, init='pca', verbose=2).fit_transform(X_train[0:5000, :])

# Calcul PCA (Pour la comparaison)
X_embedded_PCA = PCA(n_components=2, svd_solver='full').fit_transform(X_train[0:5000, :])

# Visualisation T-SNE
convex_hulls_t = convex_hulls(X_embedded, y_train[0:5000])
ellipses_t = best_ellipses(X_embedded, y_train[0:5000])
nh_t = neighboring_hit(X_embedded, y_train[0:5000])

visualization(X_embedded, y_train[0:5000], convex_hulls_t, ellipses_t, 't-SNE', nh_t)

# Visualisation PCA
convex_hulls_PCA = convex_hulls(X_embedded_PCA, y_train[0:5000])
ellipses_PCA = best_ellipses(X_embedded_PCA, y_train[0:5000])
nh_PCA = neighboring_hit(X_embedded_PCA, y_train[0:5000])

visualization(X_embedded_PCA, y_train[0:5000], convex_hulls_PCA, ellipses_PCA, 'PCA', nh_PCA)
