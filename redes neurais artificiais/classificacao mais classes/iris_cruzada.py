import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense # atualizado: tensorflow==2.0.0-beta1
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras import backend as k # atualizado: tensorflow==2.0.0-beta1

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede(): # atualizado: tensorflow==2.0.0-beta1
    k.clear_session()
    classificador = Sequential([
    tf.keras.layers.Dense(units=4, activation = 'relu', input_dim=4),
    tf.keras.layers.Dense(units=4, activation = 'relu'),
    tf.keras.layers.Dense(units=3, activation = 'softmax')])
    classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criar_rede,
                                epochs = 1000,
                                batch_size = 10)
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()
