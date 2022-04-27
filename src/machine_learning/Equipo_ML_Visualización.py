from matplotlib import pyplot
import plotly.graph_objs as go
from plotly.offline import iplot
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, auc, roc_curve
from itertools import cycle
from sklearn.preprocessing import label_binarize
import seaborn as sns
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, auc, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn import metrics




""" FUNCIÓN PARA CREAR UN SCATTERPLOT PARA ML (Dave) 

 1). El primer argumento o (dataset) es el nombre de la variable del dataset
2). Title_y es el string del Título de la gráfica, por ejemplo (Precio del Bitcoin en el 2018) """

def scatterplot(dataset):  
    return sns.scatterplot(y_test, predictions)




""" PLOT CONFUSION MATRIX (Gretel) """

"crear una matriz de confusión del porcentaje de predicciones correctas e incorrectas hechos por un modelo de clasificación"
# >> importaciones necesarias
from numpy import np
from seaborn import sns
from sklearn.metrics import confusion_matrix
# >> código
def plot_normalized_confusion_matrix(targets_tested,targets_predicted):
    cf=confusion_matrix(targets_tested,targets_predicted)
    sns.heatmap(cf/np.sum(cf), annot=True, fmt=".2%",cmap="Blues")
# >> ejemplo de llamada función
plot_normalized_confusion_matrix(y_test,y_pred)



""" FUNCIÓN DIFERENCIAS DE PÉRDIDAS DE TRAIN Y TEST (Gonzalo) """


def plot_metrics(history, metric_name, title, ylim=5):
    '''
    ----------------------------------------------------------------------------------------------
    Muestra la diferencia entre la función de pérdidas en la muestra de "train" y "validación"

    Input: 
    "history": modelo de entrada
    "metric_name": la métrica que se quiere mostrar
    "title": título para la gráfica
    "y_lim": lo que queremos mostrar en el eje y
    ----------------------------------------------------------------------------------------------
    '''
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name],color='blue',label=metric_name)
    plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)
    plt.legend()


# ML Imágenes > mostrar imágenes de cada categoría

def mostrar_imagen_de_cada_tipo(path, numero_aleatorio, categorias, nrows, ncols, figsize=(20,5)):
    '''Devuelve una figura con una imágen representativa de cada categoría

    Argumentos: 
    path (str): directorio común donde se encuentran las imágenes de cada categoría
    numero_aleatorio (int): número entero que muestre el índice de la imágen a mostrar
    categorias (list): lista con las distintas categorías

    Retornos:
    Figura con una imágen representativa de cada categoría

    Autor: Gonzalo
    '''
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, ax in enumerate(axes.flatten()):
        # plotting all 4 images on the same figure
        image_path = os.listdir(path + '/' + categorias[i])[numero_aleatorio]
        ax.title.set_text(categorias[i])
        ax.imshow(cv2.imread(path + '/' + categorias[i] + '/' + image_path)[:,:,::-1], aspect='auto')    
    plt.show()

    return fig




# Imágenes > en ML carpeta de train de imágenes contar cuantas hay por categoría

def contar_imagenes(path, classes):
    '''Cuenta las imágenes de cada categoría que encuentra dentro del directorio indicado, el cual debe de contener 
    las imágenes clasificadas en directorios según su categoría. 

    Argumentos: 
    path (str): directorio común
    classes (list): categorías dentro de las cuales se clasifica cada imagen.

    Retorno:
    df (pandas.DataFrame): DataFrame donde cada columna corresponde a cada categoría, y el número de elementos
    por categoría

    Autor: Gonzalo
    '''
    class_count = []
    for i in classes:
        class_count.append(len(os.listdir(path + '/' + i)))
        
    df = pd.DataFrame(columns = ["Class_Name", "No of Images"])
    df['Class_Name'] = classes
    df["No of Images"] = class_count
    return df


# ML > función de pérdidas (gráfico de líneas)

def plot_metrics(history, metric_name, title=None, ylim=None):
    '''Muestra la diferencia entre la función de pérdidas en la muestra de "train" y "validación"

    Argumentos: 
    history: modelo de entrada
    metric_name: la métrica que se quiere mostrar
    title (str): título para la gráfica
    y_lim (int/float): dimensiones a mostrar en el eje y
    
    Retorno:
    Figura

    Autor: Gonzalo
    '''
    plt.figure(figsize=(12,12))
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name],color='blue',label=metric_name)
    plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)
    plt.legend()
    plt.show()

    return plt.figure

# ML > ROC curve (gráfico de líneas)

def plot_roc_curve(y_real, y_pred, numero_categorias):
  '''Muestra la curva ROC, como métrica de evaluación de un modelo

  Argumentos: 
  y_real: categorías reales a las que pertenece cada imagen.
  y_pred: cateogrías predichas por el modelo
  numero_categorias (int): numero de categorias 
  
  Retorno:
  figura

  Autor: Gonzalo
  '''
  y_real = label_binarize(y_real, classes=np.arange(numero_categorias))

  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  thresholds = dict()
  for i in range(numero_categorias):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_real[:, i], y_pred[:, i], drop_intermediate=False)
  roc_auc[i] = auc(fpr[i], tpr[i])

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(numero_categorias)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(numero_categorias):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])


  # Finally average it and compute AUC
  mean_tpr /= numero_categorias

  fpr["weigthed"] = all_fpr
  tpr["weigthed"] = mean_tpr
  roc_auc["weigthed"] = auc(fpr["weigthed"], tpr["weigthed"])

  # Plot all ROC curves
  #plt.figure(figsize=(10,5))
  plt.figure(figsize=(12,7))
  lw = 2

  plt.plot(fpr["weigthed"], tpr["weigthed"],
  label="weigthed-average ROC curve (area = {0:0.2f})".format(roc_auc["weigthed"]),
  color="navy", linestyle=":", linewidth=4,)

  colors = cycle(["red","brown", "orange", "blue"])
  for i, color in zip(range(numero_categorias), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    label=("ROC curve de:", numero_categorias[i]))

  plt.figure(figsize=(12,12))
  plt.plot([0, 1], [0, 1], "k--", lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("Ratio de falsos positivos - Recall")
  plt.ylabel("Ratio de positivos verdaderos - Precision")
  plt.title("Curva ROC")
  plt.legend()

  return plt.figure




#3. PLOT CONFUSION MATRIX (CON % EN LUGAR DE NUMEROS) - propuesta 2 de 2 - (Gretel)
# >> importaciones necesarias
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# >> código
def plot_normalized_confusion_matrix(param1,param2, figsize=(12,12)):
    """Creación de una matriz de confusión: muestra los porcentajes de predicciones positivas y negativas correctas e incorrectas realizadas por un modelo de clasificación

    Argumentos:
        param1: nombre de la variable a predecir, conocida del conjunto de datos (a menuda mencionada como "y_test")
        param2: nombre de la variable a predecir tal como ha salido de la fase de verificación del modelo, para poder compararla con los valores de param1 (a menudo mencionada como "y_pred")
        figsize (tuple): tamaño de la figura
    Retornos:
        Matriz de confusión normalizada

    """
    plt.figure(figsize=figsize)
    cf=confusion_matrix(param1,param2)
    return sns.heatmap(cf/np.sum(cf), annot=True, fmt=".2%",cmap="Blues")

# >> ejemplo de llamada función
y_test = [10,6,2,6,8,13]
y_pred = [10,5,2,6,9,14]
plot_normalized_confusion_matrix(y_test,y_pred)



'''FUNCION COMPLIACION, ENTRENAMIENTO DE MODELOS DE DEEPLEARNIG PARA REGRESION Y CLASIFICACION Y RESUMEN CON GRAFICOS FUNCION LOSS Y EVOLUCION EPOCHS LAURA'''



def check_optimizadores (modelo, optimizadores, epochs, loss, metrics, x_data, y_data, bath, callbks):
    
    
    ''' Funcion que compila y entrena un modelo con uno o varios optimizadores, a través de un bucle for y muestra dos graficos.
    graf_df_result = df_results.plot.bar() muestra la funcion de perdida, tanto para train como para la parte de validacion.
    graf_epoc muestra como han ido convergiendo el modelo con cada uno de los optimizadores, tanto para el train como
    para la validadcion.   
    
    INPUT:
    modelo (objeto keras.engine.): modelo con las capas ya definidas fuera de la funcion
    optimizadores (lista): lista con el/los optimizadores que se vayan a entrenar
    epochs (int): entero con el número de epocs
    loss (str): funcion loss que se va a fijar
    metrics (objeto keras.metrics): metrica que se va a utilizar para el entrenamiento
    x_data (np.array): los datos de train para el entrenamiento
    xy_data (np.array): los datos de la variable a predecir para el entrenamiento
    bath (int): numero de paquetes o muestras para calcular el error.
    callbaks (lista): lista con las callbacks que se van a quere utilizar en el modelo
    
   
    Autor: Laura
    '''
    
    results = {}
    history = {}

    for optimizadores in optimizadores:
    
  
    
        modelo.compile (loss = loss, optimizer= optimizadores, metrics = metrics)
        optimizer_key = str(type(optimizadores).__name__)
        history[optimizer_key] = modelo.fit(x_data, y_data, batch_size = bath, epochs = int(epochs), validation_split = 0.2,callbacks= callbks)
        results[optimizer_key] = {}
        results[optimizer_key]['loss'] = history[optimizer_key].history['loss'][epochs -1]
        results[optimizer_key]['val_loss'] = history[optimizer_key].history['val_loss'][epochs -1]
        
        df_results = pd.DataFrame(results)
    graf_df_result = df_results.plot.bar()
    plt.show()
    
    graf_epoc = plt.figure(figsize= (7,7))
    plt.xlabel ('Epoch')
    plt.ylabel('Loss')
    
    for optimizadores in history:
        hist = pd.DataFrame(history[optimizadores].history)
        
        plt.plot(history[optimizadores].epoch, np.array(hist['loss']),
                label = 'Train loss' + optimizadores)
        plt.plot(history[optimizadores].epoch, np.array(hist['val_loss']),
                label = 'Val loss' + optimizadores)
    plt.legend()
    plt.show()
    
    return graf_df_result, graf_epoc



# Alfonso
# Report performance_PrettyTable
def sweet_table(X_test, y_test, *arbitrarios):
   """
    Nos prporciona una pequeña descripción de las principales métricas a utilizar par evaluar el rendimiento
    de nuestro modelo de ML. Siempre y cuando se siga el siguiente proceso: 
    1) X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
    2) Con nuestro modelo definido (ejemplo):
       model = LGBMRegressor()
       model1 = LinearRegression()
    3) Entrenado nuestro modelo:
       model.fit(X_train, y_train)
       model1.fit(X_train, y_train)
    Argumentos:
      X_test (np.array): (Ver Descripción)
      y_test (np.array): (Ver Descripción)
      *arbitrareos (str): Serán uno o varios algoritmos con los que se quiere entrenar y evaluar nuestro modelo de ML.
   """
   names = ['Metrics']
   maes = ['MAE']
   mses = ['MSE']
   rmses = ['RMSE']
   score_test = ['Accuracy (R^2)']
   # score_train = ['Accuracy (TRN)']
   # mean_rmses = ['Mean(RMSE)_CrossValidation']

   for model in arbitrarios:
      names.append(str(model))
      MAE = metrics.mean_absolute_error(y_test, model.predict(X_test))
      maes.append(str(MAE))
      MSE = metrics.mean_squared_error(y_test, model.predict(X_test))
      mses.append(str(MSE))
      RMSE = np.sqrt(metrics.mean_squared_error(y_test, model.predict(X_test)))
      rmses.append(str(RMSE))
      ACC =  metrics.r2_score(y_test, model.predict(X_test))
      score_test.append(str(ACC))
      # SCORE_TR = model.score(X_train, y_train)
      # SCORE_TS = model.score(X_test, y_test)

   x = PrettyTable()
   x.field_names = names
   x.add_row(maes)
   x.add_row(mses)
   x.add_row(rmses)
   x.add_row(score_test)
   # x.add_row(score_train)
   # x.add_row(mean_rmses)

   return x


# sweet_table(X_test, y_test, model, model1)
