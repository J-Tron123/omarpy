# -------------------------------------------------------------------------------
# IMPORTAMOS LIBRERÍAS
# -------------------------------------------------------------------------------
import os
import math as mt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import cm
from matplotlib import colors
from itertools import cycle
from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
from sklearn.preprocessing import label_binarize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.metrics import recall_score, accuracy_score, auc, roc_curve
import plotly.io as pio
pio.renderers
pio.renderers.default = "notebook_connected"

import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------
# IMPORTAMOS LIBRERÍAS
# -------------------------------------------------------------------------------


# FUNCION GRAFICO DE BARRAS HORIZONTALES
# -------------------------------------------------------------------------------

def graf_bar_horizon(eje_x, eje_y,  etiq_y, etiq_x, tittle, color = 'green',):

    ''' Funcion que muestra el conteo de las variables del eje_y y lo muestra de manera horizontal

    Argumentos:
        eje_x (lista/pandas.series): valores que queremos que tome el eje_x, 
        eje_y (lista/array/pandas.series): valores sobre los que queremos que se haga el conteo
        etiq_y (str): etique del eje y
        etiq_x (str): etique del eje x
        tittle (str): el titulo del grafico
        color (str): el color que queremos que tengan las barras del grafico, si no se indica nada por default sera green
       
    Retorno:
        figura

    Autor: Laura
    '''

    plt.figure(figsize= (7,7))
    plt.barh(eje_x, eje_y, color= color)
    plt.ylabel(etiq_y)
    plt.xlabel(etiq_x)
    plt.title(tittle)
    return plt.show()


# FUNCIÓN LINEAL
# -------------------------------------------------------------------------------
def f1(x):
    '''Función que retorna una figura con una gráfica lineal de los valores aportados en el eje X
    
    Argumentos:
        X (tuple): debe de ser un range en el que se incluya el rango de los valores de X
        
     Retorno:
        figura
        
    Autor: Dave
    '''
    return 2*(x**2) + 5*x - 2
 

# FUNCIÓN PARA GRÁFICO GENERAL DE EJE X E Y
# -------------------------------------------------------------------------------

def move_spines(x, y):
    """Esta funcion divide pone al eje y en el valor 0 de x para dividir claramente los valores positivos y
    negativos.
    
    Argumentos:
        x (int): valor del eje y
        y (int): valor del eje x
        
    Retorno:
        figura
        
    Autor : Dave 
    """
    fix, ax = plt.subplots()
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_position("zero")
    for spine in ["right", "top"]:
        ax.spines[spine].set_color("none")
    ax.grid()
    ax.plot(x,x + 3)
    plt.title(r"Grafico de $f(x)=\sqrt{x + 2}$")
    plt.ylabel('f(x)')
    plt.xlabel('x')
    plt.show()
    return ax


# Pairplot
# -------------------------------------------------------------------------------

def plot_pairplot(dataset, title_y):
    """ Creación de pairplot
    
    Argumento:
        dataset (pandas dataframe): es el nombre de la variable del dataset
        Title_y (str): Título de la gráfica, el cual debe coincidir con el nombre de la columna
    Retorno:
        figura
        
    Autor: Dave
    """
    fig = px.box(dataset, y = title_y)
    return fig.show()
   
    
# FUNCION GRAFICO DE BARRAS APILADAS PARA TRES VALORES
# -------------------------------------------------------------------------------

def stacked_bar_plot ( valores_x,data1,data2,data3,labels, ancho_barras, y_etiqueta,titulo ):

    ''' Funcion que muestra, mediante barras apliladas las frecuencias de diferentes categorias

    Argumentos:
        valores_x (lista/array/pandas.series): valores que queremos que tome el eje_x
        data1 (lista/pandas.series): categoria 1 obre los que queremos que se muestre la frecuencia
        data2 (lista/pandas.series): categoria 2 obre los que queremos que se muestre la frecuencia
        data3 (lista/pandas.series): categoria 3 obre los que queremos que se muestre la frecuencia
        labels (lista): lista de variables que queremos que se representen en la leyenda
        ancho_barras (int): ancho que queremos que tenga las barras
        y_etiqueta (str): titulo del eje y
        titulo (str): titulo del grafico
    Retorno:
        figura
        
    Autor: Laura

    '''
    plt.figure(figsize=(15,7))
    x_variable= list(range(len(valores_x)))
    indice = np.arange(len(x_variable))
    plt.bar(indice, data1, label = labels[0], width =ancho_barras, color= 'DarkOliveGreen', edgecolor= 'Gray')
    plt.bar(indice, data2, label = labels[1],  width=ancho_barras, bottom= data1,color= 'Goldenrod', edgecolor ='Gray')
    plt.bar(indice, data3, label = labels[2], width = ancho_barras,bottom= np.array(data1)+np.array(data2), color = 'DarkKhaki', edgecolor = 'Gray')
    plt.xticks(indice,valores_x)
    plt.ylabel (y_etiqueta, size= 15)
    plt.title (titulo, fontdict= {'color': 'white', 'weight': 'bold', 'size': 16 }, loc= 'center', pad=20, alpha= 0.)
    plt.grid (True, alpha=0.1)
    plt.legend()
    return plt.show()
   

# HISTOGRAMA & BOXPLOT
# -------------------------------------------------------------------------------

def num_plot(df, col, title=None, symb=None):
    '''Figura que muestra la distribución de una variable en un histograma y en un boxplot.
    Además, muestra la media, la moda y la mediana.

    Argumentos:
        df (pandas.DataFrame): pandas DataFrame
        col (str): columna del pandas DataFrame de la cual se quiere conseguir la distribución
        title (str): título del gráfico
        symb (str): unidades de medida en la que se expresa la variable

    Retorno:
        figura

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
    return fig


# HORIZONTAL BAR PLOT
# -------------------------------------------------------------------------------

def bar_hor(df, a, b, title=None, xlabels=None):
    '''Función que nos permite comparar diferentes valores.        

    Argumentos: 
        df (pandas.DataFrame) 
        a: pd.DataFrame donde la columna ['a'] sea el nombre de las variables a comparar. Puede sere el índice del dataframe. En tal caso, indicar: df.index.
        b (int / float): columna de pd.DataFrame - df['b'] - cuyos valores sean los valores de las variables a comparar.
        title (str): título que se le desea poner al gráfico. Por defecto será None.
        xlabels (str): descripción que se le quiere dar al eje X. Por defecto será None.

    Retornos:
        Figura

    Autor: Gonzalo
    '''
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    fig, ax = plt.subplots(figsize=(5,3.5))
    plt.hlines(df.a, xmin=0, xmax=df.b, color='#007acc', alpha=0.5, linewidth=5)
    plt.xticks(rotation=90)
    plt.plot(df.b, df.a, "o", markersize=5, color='#007acc', alpha=0.9)
    plt.title(title)
    plt.xlabel(xlabels)
    plt.show()
    return fig


# GRÁFICA DE BARRAS
# -------------------------------------------------------------------------------

def plot_bar_chart_with_numbers_y(param1, param2, param3, param4):
    """Creación de bar chart para 2 variables numéricas
    
    Argumentos:
        param1 (pandas dataframe): dataframe 
        param2 (str): nombre de la variable escogida
        param3 (str): nombre de la variable a predecir
        param4 (str): nobmre del gráfico

    Retornos:
        Gráfico de barras que muestra el número de valores de la variable a predecir por cada valor de la variable escogida
    
    Autor: Gretel
    """

    trace = go.Bar(x = param1[param2],
               y = param1[param3],
               marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                            line = dict(color='rgb(0,0,0)', width = 1.5)),
               text = param1[param3])
    data = [trace]
    layout = go.Layout(barmode = "group", title = param4)
    fig = go.Figure(data = data, layout = layout)
    return iplot(fig)


# BOXPLOT
# -------------------------------------------------------------------------------

def plot_boxplot(param1, param2):
    """Creación de un diagrama de caja y bigotes para representar los distintos valores de una variable numérica

    Argumentos:
        param1 (pandas dataframe): datos
        param2 (str): nombre de la variable cuya dispersión de valores se quiere pintar

    Retornos:
        Diagrama de caja y bigotes
   
    Autor: Gretel
    """
    fig = px.box(param1, param2)
    return fig.show()


# DATA REPORT
# -------------------------------------------------------------------------------

def data_report(df):
    '''Función para crear un data frame con estadísticas rápidas del dataset.
        
        Argumentos:
            df (pandas Dataframe): datos
            
        Retornos:
            DataFrame con estadísticas:
            Cols: Nombre de las columnas
            types: tipo de datos de cada columna
            percent_missing: Porcentaje de missing
            unicos: valores únicos
            percent_cardin: Porcentaje de cardinalidad
    '''
    
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)
    
    return concatenado


# GRÁFICO DE BARRAS
# -------------------------------------------------------------------------------

def grafico(df,col1,col2,titulo_x,titulo_graf):
    '''Función para representar un gráfico de barras.
       Argumentos:
            df: Nombre del dataframe
            col1 (str): Nombre de la columna para agrupa
            col2 (str) : Nombre de la segunda columna por la que agrupar
            título_x (str): título para el eje de las x
            título_graf (str): título para el gráfico
    
        Retornos:
            Gráfico de tipo barras
        
    '''
    graf = df.groupby(col1)[[col2]].count()
    trace1 = {
    'x': graf.index,
    'y': graf[col2],
    'type': 'bar'
    };
    data = [trace1];
    layout = {
    'xaxis': {'title': titulo_x},
    'title': titulo_graf
    };
    fig = go.Figure(data = data, layout = layout)
    fig.update_xaxes(
        tickangle = 90)
    return iplot(fig)


# GRÁFICO DE DISPERSIÓN
# -------------------------------------------------------------------------------

def scat_log_visualize(figuresize=(10,10), xlim=(10,10), ylim=(10,10), xlabel="X", ylabel="Y", x=np.array, y=np.array):
    '''Función que genera un gráfico de dispersión con escala logaritmica.
    
        Argumentos:
             figuresizetuple (tuple): Tamaño de la figura (default = (10,10)).
             xlim (tuple): Limites de eje X (default = (10,10)).
             ylim (tuple): Limites de eje Y (default = (10,10)).
             xlabel (str): Nombre de eje X(default = "X").
             ylabel (str): Nombre de eje Y(default = "Y").
             x (np.array): Valores en eje X.
             y (np.array): Valores en eje Y.
             
        Retorno
            Figura
    '''

    sns.set_style("whitegrid")
    sns.set_context("talk")
    plt.figure(figsize=figuresize)
    plt.axes(xscale="log", yscale="log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    sns.color_palette("hls", 8)
    sns.scatterplot(x,
                    y,
                    s=100)
    return plt.show()
   

# MAPA DE CALOR DE CORRELACIÓN
# -------------------------------------------------------------------------------

def sweet_heatmap(dataframe, figsize):
    """
    Trazar las correlaciones entre variables a través de una matriz codificada por colores.
       
    Argumentos:
      dataframe (df): (Ver Descripción)
      figsize (tuple): Tupla con el tamaño con el que se quiere dibujar el diagrama (x,x).
    
    Retorno:
        figura
    
    Autor
        Alfonso
    """
    # plot a heatmap with annotation
    plt.figure(figsize=figsize)
    sns.heatmap(dataframe.corr(),
                vmin=-1,
                vmax=1,
                cmap=sns.diverging_palette(145, 280, s=85, l=25, n=7),
                square=True,
                linewidths=.1,
                annot=True);
    return plt.show()



# PIE CHART 1
# -------------------------------------------------------------------------------

def sweet_pie_1(values, labels, title):
    """
    Un gráfico circular dividido en sectores que representan cada uno una proporción del todo.
    
    Argumentos:
      values (list_of_int): Lista con los valores numericos que se quieren representar.
      labels (list_of_str): Lista con los valores categoricos que se quieren representar.
      title (str): Titulo del grafico
      
    Retorno:
        Figura
    
    Autor: Alfonso
    """
    colors = sns.color_palette('pastel')
    explode =  [0] * len(values)
    fig, ax = plt.subplots()
    ax.pie(values, labels = labels,
            colors = colors,
            autopct='%.0f%%',
            explode = explode,
            shadow = True,
            startangle = 180)

    ax.set_title(title, weight='bold')
    return plt.show()


# PIE CHART 2
# ----------------------------------------------------------------------------
def sweet_pie_2(values, labels, title):
    """
    Un gráfico circular dividido en sectores que representan cada uno una proporción del todo.
      
    Argumentos:
      values (list_of_int): Lista con los valores numericos que se quieren representar.
      labels (list_of_str): Lista con los valores categoricos que se quieren representar.
      title (str): Titulo del grafico
    
    Retorno:
        figura
    
    Autor: Alfonso
    """
    colors = sns.color_palette('pastel')
    explode =  [0] * (len(values)-1) + [0.1]

    fig, ax = plt.subplots()
    ax.pie(values, labels = labels,
            colors = colors,
            autopct='%.0f%%',
            explode = explode,
            shadow = True,
            startangle = 180)
    ax.set_title(title, weight='bold')
    plt.axis("equal")
    return plt.show()


# BOXPLOT & DENSITY PLOT BINARIA
# ----------------------------------------------------------------------------

def plot_numerical_huetarget(data, feature):
    """Plots the boxplot and histogram of a numerical feature for each label

    Argumentos:
        data (pd.DataFrame): dataframe with the data
        feature (str): name of the feature to plot
    
    Autor: Paco
    """
    sns.set_style('whitegrid')
    palette = sns.color_palette()
    blue, green, red = palette[0], palette[2], palette[3]
    fig, axes = plt.subplots (4,1, figsize = (12,12), sharex=True, gridspec_kw={"height_ratios": (.1, .4, .1, .4)})
    sns.boxplot(x = data.query("target == 1")[feature], color = red, ax=axes[0])
    sns.histplot(data.query("target == 1")[feature], color = red, ax=axes[1])
    sns.boxplot(x = data.query("target == 0")[feature], color = green, ax = axes[2])
    sns.histplot(data.query("target == 0")[feature], color = green, ax=axes[3])
    plt.show()


# WORDCLOUD
# ----------------------------------------------------------------------------

def sweet_cloud(path, text):

    """
    Una nube de palabras es una representación visual de datos de texto. 
    Las palabras suelen ser palabras sueltas y la importancia de cada una se muestra con el tamaño o el color de la fuente.
    
    Argumentos:
      path (str): Se necesita especificar la ruta donde se encuentar laa imagen que queremos convertir.
      text (str): Necesitamos de un texto '''Texto'''
    
    Retorno:
        Figura
    
    Autor: Alfonso
    """
    stopwords = set(STOPWORDS)
    mask = np.array(Image.open(path))
    mask[mask == 1] = 255
    wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=1000, mask=mask,
    contour_width=1, contour_color='grey', min_font_size=5).generate(text)
    # create image
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return plt.show()


# PALABRAS A COLOR
# ----------------------------------------------------------------------------
def escribir_a_color(color, texto):
    """ Esta función te devuelve un string al color que tú le especifiques. 

    Argumentos: 
        color: verde, rojo, amarillo
        texto: str
        
    Retorno:
        figura
        
    Autor: Lorenzo
    """
    OK = '\033[92m' #GREEN
    WARNING = '\033[93m' #YELLOW
    FAIL = '\033[91m' #RED
    RESET = '\033[0m' #RESET COLOR

    if color == 'verde':
        resultado = OK + texto + RESET
    elif color == 'rojo':
        resultado = FAIL + texto + RESET
    elif color == 'amarillo':
        resultado = WARNING + texto + RESET

    return resultado


# GRÁFICO DE LÍNEAS
# ----------------------------------------------------------------------------

def plot_2dline(param1, param2,param3,param4=None):
    """Creación de un gráfico de línea de 2 dimensiones para representar alguna evolución en los valores numéricos de una variable
    Argumentos:
        param1 (pandas dataframe): datos
        param2 (str): nombre de la variable temporal
        param3 (str): nombre de la variable cuya evolución se quiere medir
        param4 (str): nombre del grafico de linea

    Retornos:
        Figura
    
    Autor: Gretel
    """
    fig = px.line(param1, x= param2, y= param3, title = param4)
    fig.update_xaxes(tickangle=45)
    return fig.show()
