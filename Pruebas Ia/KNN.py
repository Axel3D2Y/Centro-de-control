
import numpy as np
from collections import Counter

# Formula de la distancia euclideana
def distancia_euclideana(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)

class KNN:
    '''
    Iniciamos el numero de vecinos mas cercanos con k este nos dira cuantos vecinos vamos a tomar en la votacion para la clase
    '''
    def __init__(self, k=3): #podemos definir k con cualquier natural
        self.k= k


    '''
    Pasamos el conjunto de entrenamiento X_train y el conjunto de sus respectivas etiquetas y_train
    '''
    def fit(self,X , y): 
        self.X_train = X #Conjunto de entrenamiento
        self.y_train = y #Etiquetas del conjunto de entrenamiento


    '''
    Para cada dato en en el conjunto de entrenamiento x, lo mandamos a la funcion predicciones que nos regresa una lista 
    con la prediccion de ese dato con el modelo la salida final del modelo 
    '''

    def predict(self, X):
        predict_labels = [self._predict(x) for x in X] # cada dato x se pasa a la funcion _predict
        return np.array(predict_labels) # retorna la lista con todas las etiquetas para cada dato x
    

    '''
    Esta funcion se divide en 3 partes,
    1. calcular la distancia de cada punto x a todos los demas puntos
    2. Ordenar los datos de acuerdo a los vecinos mas cercanos 
    3. Obtener las etiquetas de los vecinos mas cercanos y votar por cual es la clase mas frecuente (por votar estamos considerando contar cual
    es la clase mas frecuente de los k vecinos cercanos)
    '''

    def _predict(self,x):
        #calculamos las distancias para cada dato x con la distancia a todos los demas puntos en el conjunto de entrenamiento  
        distancias = [distancia_euclideana(x,self.x_train) for x_train in self.X_train]

        #obtenemos los k vecinos mas cercanos, simplemente ordenando la distancia del mas cercano a mas lejado   
        k_indices = np.argsort(distancias)[:self.k] #nos devuelve una lista con los indices del orden, pero solo tomamos en consideracion los primeros k por eso el [:self.k] 
        k_vecinos_cercanos = [self.y_train[i] for i in k_indices] # Obtenemos una lista de las K etiquetas, de la clase de cada uno de los vecinos mas cercanos 

        # Preguntamos a los vecinos mas cercanos cual es la clase mas común.
        mas_comun = Counter(k_vecinos_cercanos).most_common(1) 
        '''
        Esto nos devuelve una lista de varias listas 
        que contiene una tupla (dato mas cercano, frecuencia) para cada dato comparado con cada clase del conjunto X_train

        '''
        return mas_comun[0][0] #nos quedamos unicamente con la clase mas comun para el punto x tomando solo la respuesta de los k vecinos mas cercanos


