"""
TP 2 - Deep Learning avec Keras et Manifold Untangling
Exercice 1 : Régression Logistique avec Keras
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils

# Création du modèle
model = Sequential()

# Ajout des couches cachées
model.add(Dense(10, input_dim=784, name='fc1'))
model.add(Activation('softmax'))

# Taux d'apprentissage
learning_rate = 0.1
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

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

# Apprentissage
batch_size = 100
nb_epoch = 100
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluation du modèle
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
