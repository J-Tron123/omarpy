

# VISUALIZACIÓN
# BARPLOT SIMPLE
import matplotlib.pyplot as plt

def bar_hor(df, title=None, xlabels=None):
    '''
    Función que nos permite comparar diferentes valores.        

    INPUTS: 
    df.a: pd.DataFrame donde la columna ['a'] sea el nombre de las variables a comparar.
    df.b: pd.DataFrame donde la columna ['b'] sean los valores de las variables a comparar. Deben ser tipo float o int. 
    title: título que se le desea poner al gráfico. Por defecto será None.
    xlabels: descripción que se le quiere dar al eje X. Por defecto será None.

    OUTPUT:
    Gráfico de barras horizontales

    Artista / culpable: Gonzalo
    '''
    # Set styles for axes
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'

    # Plot
    fig, ax = plt.subplots(figsize=(5,3.5))
    plt.hlines(df.a, xmin=0, xmax=df.b, color='#007acc', alpha=0.5, linewidth=5)

    plt.xticks(rotation=90)
    plt.plot(df.b, df.a, "o", markersize=5, color='#007acc', alpha=0.9)
    plt.title(title)
    plt.xlabel(xlabels)
    plt.show()


# ML: CNN de clasificación
import os
import cv2

def mostrar_imagen_de_cada_tipo(path, numero_aleatorio, categorias):
    '''
    -----------------------------------------------------------------------------------------
    Devuelve una figura con una imágen representativa de cada categoría

    Input: 
    "path": directorio común donde se encuentran las imágenes de cada categoría
    "numero_aleatorio": número (dtype: int) que muestre el índice de la imágen a mostrar
    "categorias": lista con las distintas categorías

    Output:
    Figura con una imágen representativa de cada categoría
    -----------------------------------------------------------------------------------------
    '''
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20,5))
    for i, ax in enumerate(axes.flatten()):
        # plotting all 4 images on the same figure
        image_path = os.listdir(path + '/' + categorias[i])[numero_aleatorio]
        ax.title.set_text(categorias[i])
        ax.imshow(cv2.imread(path + '/' + categorias[i] + '/' + image_path)[:,:,::-1], aspect='auto')    
    plt.show()


# CNN ----------------------------------------------------------------------------------------------- NO ENVIADA
import numpy as np

# Tranformaciones
def define_x_y(img_folder, img_width, img_height):
    '''
    -----------------------------------------------------------------------------------------
    Devuelve una "X" con el listado de imágenes a clasificar, y una "y" con la categoría
    de cada una de ellas

    Input: 
    "img_folder": directorio común
    "img_width" y "img_height": tamaño de los pixeles de cada imagen

    Output:
    "X" e "y" en forma de lista, con valores tipo float normalizados
    -----------------------------------------------------------------------------------------
    '''
   
    X = list()
    y = list()

   # Iteramos en el directorio, para cada una de las carpetas clasificadoras, 
    for i in os.listdir(img_folder):
        new_path = img_folder + '/' + i
        for j in os.listdir(new_path): # Iteramos para alcanzar cada una de las imágenes en cada directorio, y pasarlo a un array
            image_path= new_path + '/' + j # path de cada imagen
            image= cv2.imread(image_path, cv2.COLOR_BGR2RGB) # Lee la imagen y la transforma al formato de color apropiado
            image=cv2.resize(image, (img_width, img_height),interpolation = cv2.INTER_AREA) # Da a cada imagen la dimensión indicada
            image=np.array(image)
            image = image.astype('float32') # Convierte la imagen a un numpy array, tipo float
            image /= 255 # Normalizamos la imagen a valores entre 0 y 1 (por defecto van de 0 a 255), ayudará al modelo
            X.append(image)
            y.append(i)
    
    return X, y



# Tratamiento de imágenes
import pandas as pd

def contar_imagenes(path, classes):
    '''
    -----------------------------------------------------------------------------------------
    Cuenta las imágenes de cada categoría que encuentra dentro del directorio indicado

    Input: 
    "path": directorio común
    "classes": categorías dentro de las cuales se clasifica cada imagen

    Output:
    DataFrame donde cada columna corresponde a cada categoría, y el número de elementos
    por categoría
    -----------------------------------------------------------------------------------------
    '''
    class_count = []
    for i in classes:
        class_count.append(len(os.listdir(path + '/' + i)))
        
    df = pd.DataFrame(columns = ["Class_Name", "No of Images"])
    df['Class_Name'] = classes
    df["No of Images"] = class_count
    return df


# CNN: evaluación del modelo
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, auc, roc_curve
from itertools import cycle
from sklearn.preprocessing import label_binarize


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

def plot_roc_curve(y_real, y_pred, numero_categorias):
  '''
  ----------------------------------------------------------------------------------------------
  Muestra la curva ROC, como métrica de evaluación de un modelo

  Input: 
  "y_real": categorías reales a las que pertenece cada imagen
  "y_pred": las cateogrías predichas por el modelo
  "numero_categorias": numero de categorias
  ----------------------------------------------------------------------------------------------
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

  plt.plot([0, 1], [0, 1], "k--", lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate - Recall")
  plt.ylabel("True Positive Rate - Precision")
  plt.title("Curva ROC")
  plt.legend()


# EDA ------------------------------------------------------------------ 
import seaborn as sns

def num_plot(df, col, title, symb):
    '''
    Figura que muestra la distribución de una variable en un histograma y en un boxplot.
    Además, muestra la media, la moda y la mediana.

    INPUT:
    df: pandas DataFrame
    col: columna del pandas DataFrame de la cual se quiere conseguir la distribución
    title: título del gráfico
    symb: unidades de medida en la que se expresa la variable

    Autor: Gonzalo
    '''
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,5),gridspec_kw={"height_ratios": (.2, .8)})
    ax[0].set_title(title,fontsize=18)
    sns.boxplot(x=col, data=df, ax=ax[0])
    ax[0].set(yticks=[])
    sns.histplot(x=col, data=df, ax=ax[1])
    ax[1].set_xlabel(col, fontsize=16)
    plt.axvline(df[col].mean(), color='darkgreen', linewidth=2.2, label='mean=' + str(np.round(df[col].mean(),1)) + symb)
    plt.axvline(df[col].median(), color='red', linewidth=2.2, label='median='+ str(np.round(df[col].median(),1)) + symb)
    plt.axvline(df[col].mode()[0], color='purple', linewidth=2.2, label='mode='+ str(df[col].mode()[0]) + symb)
    plt.legend(bbox_to_anchor=(1, 1.03), ncol=1, fontsize=17, fancybox=True, shadow=True, frameon=True)
    plt.tight_layout()
    plt.show()