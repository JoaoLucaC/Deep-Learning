import pandas as pd
import tensorflow as tf
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from tensorflow.keras import backend as k 
from tensorflow.keras.models import Sequential 

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(): 
    k.clear_session()
    classificador = Sequential([
               tf.keras.layers.Dense(units=16, activation = 'relu', kernel_initializer = 'random_uniform', input_dim=30),
               tf.keras.layers.Dense(units=16, activation = 'relu', kernel_initializer = 'random_uniform'),
               tf.keras.layers.Dropout(0.2),
               tf.keras.layers.Dense(units=1, activation = 'sigmoid')])
    otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, decay = 0.0001, clipvalue = 0.5)
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(model = criarRede,
                                epochs = 100,
                                batch_size = 10)
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 2, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()