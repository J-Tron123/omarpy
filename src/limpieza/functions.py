import os
import shutil
import cv2
import numpy as np
import pandas as pd
from fancyimpute import IterativeImputer as MICE

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

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

#En alguna ocasión me ha sido favorable agrupar por porcentaje.

def columnascat(df,nombrecolumntarget,split=" "):                                  #Función para tratar variables categoricas en problemas de clasificación binaria. Necesita el nombre de la columna target entrecomillada y viene por default que el split se haga con un espacio (split=" ").
    le=LabelEncoder()
    for i in df.columns:                                                           #Recorremos las columnas en busca de variables categoricas
        if df[i].dtype==("object" or str):
            print(i)
            if (len(set(df[i]))<=(len(df)/20)):                                    #Miramos si hay, por lo menos,5 variables diferentes por cada 100 registros 
                df[i]=le.fit_transform(df[i])                                      #Si los hay, hacemos un label encoder
                for q in set(df[i]):                                               #Recorremos las diferentes variables 
                    x=df[df[i]==q]
                    porcentaje= ((len(x[x[nombrecolumntarget]==1]))*100)/(len(x))  #Se relaciona la variable con el target y se saca el porcentaje de probabilidad
                    mask=(df[i]==q)
                    df.loc[mask,i]=porcentaje                                      #Se aplica una mascara para cambiar cada variable por su porcentaje de probabilidad
            
            else:
                try:
                    data=df[i].str.split(split)                                     #Si no tenemos minimo 5 variables por cada 100 registros, intentamos dividir las diferentes variables. Util cuando son marcas de coche, procesadores,etc.
                    df["New"]=data[0]

                    if (len(set(df["New"]))<=(len(df)/20)):                         #Miramos ahora si tenemos 5 variables por cada 100 registros. Si es asi, repetimos el proceso de transformación y agrupación por porcentaje         
                        df[i]=le.fit_transform(df[i])
                        for q in set(df[i]):
                            x=df[df[i]==q]
                            porcentaje= ((len(x[x[nombrecolumntarget]==1]))*100)/(len(x))
                            mask=(df[i]==q)
                            df.loc[mask,i]=porcentaje
                except:
                    print("No se puede dividir")
        else:
            print("No es una variable categorica")

def beautifull_scrap(url, headers):
    """Función que obtiene la información de una página web mediante web scrapping

    Argumentos:
        url (str): Dirección URL de la web que se quiere atacar.
        headers (dict): Diccionario con la cabecera que requiere la consulta a la web.

    Retornos:
        soup: HTML parseado de la librería bs4
    """
    import bs4 as bs, requests

    response = requests.get(url, headers=headers)
    return bs(response.content, "lxml")

def drop_missings(df, axis, limit=""):
    """Función que elimina los valores nulos de un DataFrame de la librería pandas

    Argumentos:
        df (str): DataFrame al que se limpiaran los valores nulos.
        axis (dict): Eje de donde se toma el criterio de eliminar dichos valores nulos, 
        si es 0 eliminara las filas con valores nulos, si es 1 eliminara las columnas con una 
        suma de valores nulos superior al límite.
        limit (int): Limite de valores nulos que se pueden tolerar para no eliminar una columna, por defecto es
        el 25% de la cantidad de registros que tenga el DataFrame

    Retornos:
        df: DataFrame
    """
    if axis == 1:

        if limit != ":":
            limit = len(df) * 0.25
        try:
            limit = int(limit)
        except:
            return "Limit debe ser un entero"

        for column in df.columns:
            if df[column].isnull().sum() > limit:
                df.drop(column, axis=1, inplace=True)
    else:
        df.dropna(axis=0, inplace=True)

    return df

def file_sorter(path: str):
    """
    A function to sort the files in a folder
    into their respective categories.

    Args:
        path: The destination path were to sort files.
    
    Returns:
        A print showing where the files are moved to.
    """
    
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    ft_list=[] # ft (file type)
    ft_folder_dict={}
    
    # Creating folders for each kind el file
    for file in files:
        # Storing file stype extension
        ft = file.split('.')[1]

        if ft not in ft_list:
            ft_list.append(ft)
            new_folder_name = path + '/' + ft + '_folder'
            ft_folder_dict[str(ft)] = str(new_folder_name)

            if os.path.isdir(new_folder_name) == True:  # Folder exists
                continue
            
            else:
                os.mkdir(new_folder_name)
    
    # Moving files to respectively folder
    for file in files:
        src_path = path + '/' + file
        ft = file.split('.')[1]

        if ft in ft_folder_dict.keys():
            dest_path = ft_folder_dict[str(ft)]
            shutil.move(src_path, dest_path)
            
    print(src_path + ('\033[92m' + ' >>> ' + '\033[0m') + dest_path)

def contar_imagenes(path: str, classes: list):
    '''
    Cuenta las imágenes de cada categoría que encuentra dentro del directorio indicado, el cual debe de contener 
    las imágenes clasificadas en directorios según su categoría. 

    Argumentos: 
        path: Directorio común.
        classes: Lista de categorías dentro de las cuales se clasifica cada imagen.

    Retorno:
        DataFrame donde cada columna corresponde a cada categoría, y el número de elementos
        por categoría.
    '''
    try:
        if '\\' in path:
            path = path.replace('\\', '/')
        else:
            pass

        class_count = []
        for i in classes:
            class_count.append(len(os.listdir(path + '/' + i)))
            
        df = pd.DataFrame(columns = ["Class_Name", "No of Images"])
        df['Class_Name'] = classes
        df["No of Images"] = class_count
    
        return df

    except:
        print('Usa el tipo de datos apropiados para esta función.')

def create_dict_images(directory):
    """
    Funcion que crea diccionario con el directorio completo de la imagen y la imagen.

    Args:
        directory: El directorio.
    
    Returns:
        Un diccionario de las ubicaciones de las imágenes.
    """
    image_dict = {}
   
    for filename in os.listdir(directory):
        full_address = directory + '/' + filename
        # Read image and convert the BGR image to RGB
        # save filename and image in dictionary 
        image_dict.update({filename: cv2.imread(full_address, cv2.COLOR_BGR2RGB)})

    return image_dict 

def remover_vello(imagen: np.array):
    '''
    Se aplica una máscara para eliminar el vello en la imagen. La función devuelve una `imagen` con las
    mismas dimensiones que la original.
    
    Args:
        imagen: Matríz con valores entre 0 y 255.

    Returns:
        new_img: La imagen (matríz) con los filtros aplicados.
    '''
    try:
        #1. Convertir a escala de grises
        img_grayScale = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        # Kernel para el filtrado morfológico
        kernel = cv2.getStructuringElement(1,(17,17))
        # Filtrado BlackHat para encontrar los contornos del cabello
        blackhat = cv2.morphologyEx(img_grayScale, cv2.MORPH_BLACKHAT, kernel)
        # Intensificar los contornos del cabello en preparación para el algoritmo de pintura
        _,mask = cv2.threshold(blackhat,12,255,cv2.THRESH_BINARY)
        # Pintar la imagen original dependiendo de la máscara
        new_img = cv2.inpaint(imagen,mask,1,cv2.INPAINT_TELEA)

        return new_img
    
    except:
        print(f'El formato {imagen} no es el adecuado. Revisa la descripción de la función.')

def mask_fondo(imagen: np.array):
    '''
    Función para eliminar el fondo de la imagen.

    Args:
        imagen: Matríz con valores entre 0 y 255.

    Returns:
        new_img: La imagen (matríz) con los filtros aplicados.
    '''
    try:
        gray_example = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_example, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask_inv = cv2.bitwise_not(mask) 
        new_img = cv2.bitwise_and(imagen,imagen,mask = mask_inv)

        return new_img

    except:
        print(f'El formato {imagen} no es el adecuado. Revisa la descripción de la función.')