import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
from plotly.offline import iplot
import warnings
warnings.filterwarnings("ignore")
import numpy as np


def data_report(df):
    '''Función para crear un data frame con estadísticas rápidas del dataset.
        
        Argumentos:
        - df = Dataframe

        Retornos:
        - DataFrame con estadísticas:
        - Cols: Nombre de las columnas
        - types: tipo de datos de cada columna
        - percent_missing: Porcentaje de missing
        - unicos: valores únicos
        - percent_cardin: Porcentaje de cardinalidad
        
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




def grafico(df,col1,col2,titulo_x,titulo_graf):
    '''Función para representar un gráfico de barras.

        Argumentos:
        - df: Nombre del dataframe
        - col1 (str): Nombre de la columna para agrupa
        - col2 (str) : Nombre de la segunda columna por la que agrupar
        - título_x (str): título para el eje de las x
        - título_graf (str): título para el gráfico
    
        Retornos:
        - Gráfico de tipo barras
        
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



def scat_log_visualize(figuresize=(10,10), xlim=(10,10), ylim=(10,10), xlabel="X", ylabel="Y", x=np.array, y=np.array):
    '''Función que genera un gráfico de dispersión con escala logaritmica.
    
        Argumentos:
        - figuresizetuple (tuple): Tamaño de la figura (default = (10,10)).
        - xlim (tuple): Limites de eje X (default = (10,10)).
        - ylim (tuple): Limites de eje Y (default = (10,10)).
        - xlabel (str): Nombre de eje X(default = "X").
        - ylabel (str): Nombre de eje Y(default = "Y").
        - x (np.array): Valores en eje X.
        - y (np.array): Valores en eje Y.
    '''

    sns.set_style("whitegrid")
    sns.set_context("talk")
    plt.figure(figsize=figuresize)
    plt.axes(xscale="log", yscale="log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.color_palette("hls", 8)
    sns.scatterplot(x,
                    y,
                    s=100);



def heatmap(data):
    
    """ Realizar un heatmap del dataframe que quieras

    Argumentos:
        - data: Dataframe que elijas

    """
    plt.figure(figsize=(20,20))
    correlacion = np.abs(data.corr())
    sns.set_style("ticks")
    sns.heatmap(correlacion, annot = True, cmap=plt.cm.Blues)
    plt.show()



def graphs_sub(number_r, number_c):
    
    """Crear Subplots con el número de filas y columnas dadas.

    Argumentos:
        - number_r (int): Número de filas o rows.
        - number_c (int): Número de columnas o columns.       
        
    Retornos:
        - fig: Subplots.
        - axes: axis de los subplots
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 15})    
    fig, axes= plt.subplots(number_r, number_c, figsize = (number_c * 8, number_r * 8))
    return fig, axes