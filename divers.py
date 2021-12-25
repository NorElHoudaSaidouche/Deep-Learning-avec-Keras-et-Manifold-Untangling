"""
TP 1 - Deep Learning avec Keras et Manifold Untangling
Implémentation des fonctions
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""

# Importation des bibliothèques
from keras.models import model_from_yaml


def save_model(model, savename):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(savename + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        print("Yaml Model ", savename, ".yaml saved to disk")
    # serialize weights to HDF5
    model.save_weights(savename + ".h5")
    print("Weights ", savename, ".h5 saved to disk")
