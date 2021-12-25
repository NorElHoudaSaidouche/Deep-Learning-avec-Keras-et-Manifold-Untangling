"""
TP 2 - Deep Learning avec Keras et Manifold Untangling
Exercice 5 : Visualisation des représentations internes des réseaux de neurones
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.manifold import TSNE

from divers import *

# Création de l'ensemble d'apprentissage et de test
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Conversion des lables en matrices de classe binaires
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Chargement du modèle MLP
model_mlp = load_model('modele_exo2')
model_mlp.summary()

# Taux d'apprentissage
learning_rate = 0.1
sgd = SGD(learning_rate)
model_mlp.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Evaluation du modèle
scores = model_mlp.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model_mlp.metrics_names[0], scores[0] * 100))
print("%s: %.2f%%" % (model_mlp.metrics_names[1], scores[1] * 100))

# Suppression des couches
# Couche d’activation softmax
model_mlp.pop()
# Couche complètement connectée
model_mlp.pop()
model_mlp.summary()

# Prédiction
pred_mlp = model_mlp.predict(X_test)

# T-SNE
X_embedded_mlp = TSNE(n_components=2, perplexity=30.0, init='pca', verbose=2).fit_transform(pred_mlp)

# Visualisation
convex_hulls_t_mlp = convex_hulls(X_embedded_mlp, y_test)
ellipses_t_mlp = best_ellipses(X_embedded_mlp, y_test)
nh_t_mlp = neighboring_hit(X_embedded_mlp, y_test)

visualization(X_embedded_mlp, y_test, convex_hulls_t_mlp, ellipses_t_mlp, 't-SNE_MLP', nh_t_mlp)

# Chargement du modèle CNN
model_cnn = load_model('modele_exo3')
model_cnn.summary()

# Reshape des données pour CNN
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Taux d'apprentissage
learning_rate = 0.1
sgd = SGD(learning_rate)
model_cnn.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Evaluation du modèle
scores = model_cnn.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model_cnn.metrics_names[0], scores[0] * 100))
print("%s: %.2f%%" % (model_cnn.metrics_names[1], scores[1] * 100))

# Suppression des couches
# Couche d’activation softmax
model_cnn.pop()
# Couche complètement connectée
model_cnn.pop()
model_cnn.summary()

# Prédiction
pred_cnn = model_cnn.predict(X_test)

# T-SNE
X_embedded_cnn = TSNE(n_components=2, perplexity=30.0, init='pca', verbose=2).fit_transform(pred_cnn)

# Visualisation
convex_hulls_t_cnn = convex_hulls(X_embedded_cnn, y_test)
ellipses_t_cnn = best_ellipses(X_embedded_cnn, y_test)
nh_t_cnn = neighboring_hit(X_embedded_cnn, y_test)

visualization(X_embedded_cnn, y_test, convex_hulls_t_cnn, ellipses_t_cnn, 't-SNE_CNN', nh_t_cnn)
