"""
TP 2 - Deep Learning avec Keras et Manifold Untangling
Exercice 3 : Réseaux de neurones convolutifs avec Keras
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
from divers import save_model
from keras.layers import Conv2D, MaxPooling2D

# Création de l'ensemble d'apprentissage et de test
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Conversion des lables en matrices de classe binaires
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Création du modèle
model = Sequential()

# Ajout des couches

model.add(Conv2D(16, kernel_size=(5, 5), input_shape=(28, 28, 1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(28, 28, 1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('sigmoid'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

# Taux d'apprentissage
learning_rate = 0.1
sgd = SGD(learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Apprentissage
batch_size = 100
nb_epoch = 100
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluation du modèle
scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Sauvegarde du modèle
save_model(model, 'modele_exo3')
