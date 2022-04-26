import os
import cv2
import numpy as np
import pandas as pd



def num_describe(data_in):
    """returns a better vesion of describe

    Args:
        data_in (pd.DataFrame): input pandas DataFrame

    Returns:
        pd.DataFrame: output dataframe
    """
    # get extra percentiles
    data_out = data_in.describe([.01,.02,.98,.99]).T
    data_out = data_out.drop(columns='count')
    data_out.insert(0,'skewness', data_in.skew())
    data_out.insert(0,'kurtosis', data_in.kurtosis())
    data_out.insert(0,'sparsity', (data_in==0).sum()/len(data_in))
    data_out.insert(0,'nulls', (data_in.isna()).sum()/len(data_in))
    return data_out


def read_images(path,size,filter=None):
    '''
    Función para cargar las imágenes en un array. 

    Args:
        path : str
        size : tuple
        filter : funct (Default = None)

    Returns:
        X : array
    '''
    X = []
    for img in os.listdir(path):
        image = cv2.imread(path+"/"+img)
        if filter !=None:
            image = filter(image)
        smallimage = cv2.resize(image, size)
        X.append(smallimage)
    return np.array(X)
def circ_distance(value_1, value_2, low, high):
    """Returns distance bweteen two cyclical values

    Args:
        value_1 (int,float): first value
        value_2 (int,float): second value
        low (int,float): _description_
        high (int,float): _description_

    Returns:
        float: distance between two values
    """
    # minmax scale to 0-2pi rad
    value_1_rad = ((value_1-low)/(high-low))*(2*np.pi)
    value_2_rad = ((value_2-low)/(high-low))*(2*np.pi)
    # sin and cos for coordinates in the unit circle 
    sin_value_1, cos_value_1 = np.sin(value_1_rad), np.cos(value_1_rad)
    sin_value_2, cos_value_2 = np.sin(value_2_rad), np.cos(value_2_rad)
    # dot product is the arccos of alpha
    angle = np.arccos(np.dot([cos_value_1, sin_value_1],[cos_value_2, sin_value_2]))
    # convert back to initial units
    angle = angle*(high-low)/(2*np.pi) + low
    # return distance
    return round(angle,3)


def inf_as_nan(df=pd.DataFrame):
    """Remplaza valores infinitos de un DataFrame por NaN para poder operar con ellos.
        
        Argumentos:
        df_column = Columna de dataframe. 

        
    """

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
def mice_impute_nans(df):
    ''' función que utiliza la Imputación Múltiple por Ecuaciones Encadenadas
        function which uses Multiple Imputation by Chained Equation
    
        argumentos:
        dataFrame = datos sin valores que faltan 
        arguments: 
        dataframe = dataframe without missing values''' 

    cols = df.columns
    imputed_df = MICE( min_value= 0).fit_transform(df.values)
    imputed_df = pd.DataFrame(imputed_df)
    imputed_df.set_index(df.index, inplace=True)
    imputed_df.columns= cols

    return imputed_df


def remove_units (DataFrame, columns, units):
    """Eliminar algunas extensiones, por ejemplo unidades de medida;
    se puede incluir cuantas columnas sea necesario;
    La variable "units" sería, por ejemplo, "Kg", "mol/L" """
    for col in columns:
        DataFrame[col] = DataFrame[col].str.strip(units)
 


def convertidor_español(Dataframe, column_name):
    """ Conversión de números en formato string a float cuando estos tienen la puntuación 
    no anglosajona.
    Parámetros:
        - Dataframe: dataframe
        - Column_name: str"""

    Dataframe[column_name] = Dataframe[column_name].str.replace(".","")
    Dataframe[column_name] = Dataframe[column_name].str.replace(",",".").astype(float)

    return Dataframe


def convertidor_ingles(Dataframe, column_name):
    """ Conversión de números en formato string a float cuando estos tienen la puntuación 
    anglosajona.
    Parámetros:
        - Dataframe: dataframe
        - Column_name: str"""

    Dataframe[column_name] = Dataframe[column_name].str.replace(",","").astype(float)
    return Dataframe

def normalize(palabra):
    """ Normalización de la palabras quitando símbolos de acentuación

    Parámetros:
        - palabra: str"""

    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for a, b in replacements:
        palabra = palabra.replace(a, b).replace(a.upper(), b.upper())
    return palabra




def crear_rentabilidades(Dataframe,Column_precio, nombre_nueva_columna, n): 

    """ Esta función se emplea para calcular la rentabilidad de un precio n períodos atras.
    Se crea una nueva columna.

    Parametros: 
        Dataframe: Df
        Column_precio: str
        Nombre_columna: Str
        n: int.  """

    lista_precios_n_periodos_atras = []

    for x,i in enumerate (Dataframe[Column_precio]):
        if x + n >= len(Dataframe[Column_precio]):
            lista_precios_n_periodos_atras.append(np.nan)
        else: 
            lista_precios_n_periodos_atras.append(Dataframe[Column_precio][x+1])
    Dataframe['PRECIO ANTERIOR']=lista_precios_n_periodos_atras
    Dataframe[nombre_nueva_columna]=(Dataframe[Column_precio]-Dataframe['PRECIO ANTERIOR'])/Dataframe['PRECIO ANTERIOR']


    return Dataframe