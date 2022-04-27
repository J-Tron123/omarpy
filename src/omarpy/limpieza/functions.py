import pandas as pd 
import numpy as np
import regex as re  
from sklearn.model_selection import train_test_split
import string
from sklearn.preprocessing import LabelEncoder
import os
import cv2
from fancyimpute import IterativeImputer as MICE
import bs4 as bs, requests


def drop_missings(df, axis, limit=""):
    """
    Función que elimina los valores nulos de un DataFrame de la librería pandas

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

def remove_units (DataFrame, columns, units):
    """
    Eliminar algunas extensiones, por ejemplo unidades de medida;
    se puede incluir cuantas columnas sea necesario;
    La variable "units" sería, por ejemplo, "Kg", "mol/L" 
    
        Argumentos:
        - DataFrame: Nombre del dataframe
        - columns (str): nombre de columna
        - units (str) : string que queramos eliminar
    
    """
    for col in columns:
        DataFrame[col] = DataFrame[col].str.strip(units)
    return DataFrame

def to_type(DataFrame, columns, type):
    """
    Funcion para cambiar el tipo de la columna, 
    debe introducirse el nombre del Dataframe,
    el nombre de la columna que quiere cambiar y el tipo fin

        Argumentos:
        - DataFrame: Nombre del dataframe
        - columns (str): nombre de columna
        - type (str) : tipo al que queramos cambiar   
    
    """
    DataFrame[columns] = DataFrame[columns].astype(type)
    return DataFrame

def filter_df(DataFrame, columns, num):
    '''Función para crear un nuevo dataframe considerando lineas de una columna que tengan el mismo valor

        Argumentos:
        - DataFrame: Nombre del dataframe
        - columns (str): nombre de columna
        - num (int | float) : número que queramos mantener en la columna    

        Return:
        -Df con la mascara


    '''      

    df_filter = DataFrame[DataFrame[columns]==num]
        
    return df_filter

def col_to_float(df_column):
    """Retorna una columna de dataframe pandas con sus valores como float si esta compuesto por números y contiene
    carácteres como ('.'/','/' ').
        
        Argumentos:
        - df_column = Columna de dataframe. 

        Return:
        -Df modificado
        
    """

    df_column = pd.Series([x.replace(".","") for x in df_column])
    df_column = pd.Series([x.replace("-","") for x in df_column])
    df_column = pd.Series([x.replace(" ","") for x in df_column]).astype(float)

    return df_column

def nan_as_nan(df_column_to_edit, nan_value):
    """Retorna valores nulos diferentes a NaN (ej:999) como NaN (None)

        Argumentos:
        - df_column_to_edit (str) : Nombre de la columna
        - nan_value (int  | float) : lo que queramos transformar a NaN

        Return:
        -Columna modificada

    """
    df_column_to_edit = np.where(df_column_to_edit == nan_value, None, df_column_to_edit)

    return df_column_to_edit

def inf_as_nan(df):
    """Remplaza valores infinitos de un DataFrame por NaN para poder operar con ellos.
        
        Argumentos:
        - df = Dataframe.         
    """

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def regex_tex_all(raw,  pattern_b = "", pattern_a = "", special_characters = "", white_space = False):
    """De acuerdo al patrón dado, extraer todas las ocurrencias de un texto o lista de textos.
    
    Argumentos:
        - raw (str): Texto o lista de texto a extraer la información.
        - pattern_b (str): Patrón antes de la palabra o cadena de palabras deseada,
                        incluir espacios para ser más precisa la búsqueda; default = "".
        - pattern_a (str): Patrón después de la palabra o cadena de palabras deseada, 
                        incluir espacios para ser más precisa la búsqueda; default = "".
        - special_characters (str): Agregar si la información a obtener tiene caracteres
                        especiales como punto(.), coma(,), guión(-), entre otros; default = "".
        - white_space (bool): True si la cadena de palabras tienen espacios, y False de lo contrario, default = False.
        
    Retornos:
        - list: Lista con la información encontrada.
    """ 
    if type(raw) == str:
        raw = [raw]
        
    list_words = []
    for word in raw:
        if white_space == True:
            
            if pattern_b == "":
                re_string = re.findall(f"([\w\s{special_characters}]+){pattern_a}", word)
            elif pattern_a== "":
                re_string = re.findall(f"{pattern_b}([\w\s{special_characters}]+)", word)
            else:
                re_string = re.findall(f"{pattern_b}([\w\s{special_characters}]+){pattern_a}", word)
                            
        else:
            if pattern_b == "":
                re_string = re.findall(f"([\w{special_characters}]+){pattern_a}", word)
            elif pattern_a== "":
                re_string = re.findall(f"{pattern_b}([\w{special_characters}]+)", word)
            else:
                re_string = re.findall(f"{pattern_b}([\w{special_characters}]+){pattern_a}", word)        
        list_words.append(re_string)
    
    return list_words

def regex_tex_first(raw,  pattern_b = "", pattern_a = "", special_characters = "", white_space = False):
    """De acuerdo al patrón dado, extraer la primera ocurrencia de un texto o lista de textos.
    
    Argumentos:
        - raw (str): Texto o lista de texto a extraer la información.
        - pattern_b (str): Patrón antes de la palabra o cadena de palabras deseada,
                        incluir espacios para ser más precisa la búsqueda; default = "".
        - pattern_a (str): Patrón después de la palabra o cadena de palabras deseada, 
                        incluir espacios para ser más precisa la búsqueda; default = "".
        - special_characters (str): Agregar si la información a obtener tiene caracteres
                        especiales como punto(.), coma(,), guión(-), entre otros; default = "".
        - white_space (bool): True si la cadena de palabras tienen espacios, y False de lo contrario, default = False.
        
    Retornos:
        - list: Lista con la información encontrada.
    """

    if type(raw) == str:
        raw = [raw]
        
    list_words = []
    for word in raw:
        if white_space == True:
            
            if pattern_b == "":
                re_string = re.findall(f"([\w\s{special_characters}]+){pattern_a}", word)
            elif pattern_a== "":
                re_string = re.findall(f"{pattern_b}([\w\s{special_characters}]+)", word)
            else:
                re_string = re.findall(f"{pattern_b}([\w\s{special_characters}]+){pattern_a}", word)
                            
        else:
            if pattern_b == "":
                re_string = re.findall(f"([\w{special_characters}]+){pattern_a}", word)
            elif pattern_a== "":
                re_string = re.findall(f"{pattern_b}([\w{special_characters}]+)", word)
            else:
                re_string = re.findall(f"{pattern_b}([\w{special_characters}]+){pattern_a}", word)
        try:       
            list_words.append(re_string[0])
        except:
            list_words.append(None)
    
    return list_words

def drop_rows_by_index(df_raw, column, value_to_r, comparison_operator = "==", negation = False):
    """Eliminar filas o registros de un DataFrame, de acuerdo a un filtro dado.
    
    Argumentos:
        - df_raw (pandas.DataFrame): DataFrame a modificar.
        - column (str): Columna donde se encuentra el valor o los valores a eliminar.
        - value_to_r (str, int o float): Valor a eliminar, puede ser un valor tipo string, entero o float. 
                                    Este debe ser igual al tipo de variable de la columna indicada.
        - comparison_operator (str): Incluir el operador de comparación de los valores a eliminar, por ejemplo,
                                    "!=" distinto a, ">" mayor a, "<" menor a, ">=" mayor o igual a, "<=" menor o igual a;
                                    default = "==" igual.
        - negation (bool): True agregar la virguilla(~), es decir negar el operador de comparación indicado, default = False.
        
    Retornos:
        - pandas.DataFrame: DataFrame modificado.
    """
    df = df_raw.copy()
    
    if negation == False:
        if comparison_operator == "==":
            indexes = df[df[column] == value_to_r].index
        elif comparison_operator == ">":
            indexes = df[df[column] > value_to_r].index
        elif comparison_operator == ">=":
            indexes = df[df[column] >= value_to_r].index        
        elif comparison_operator == "<":
            indexes = df[df[column] < value_to_r].index        
        elif comparison_operator == "<=":
            indexes = df[df[column] <= value_to_r].index        
        else:
            indexes = df[df[column] != value_to_r].index
    
    else:
        if comparison_operator == "==":
            indexes = df[~(df[column] == value_to_r)].index
        elif comparison_operator == ">":
            indexes = df[~(df[column] > value_to_r)].index
        elif comparison_operator == ">=":
            indexes = df[~(df[column] >= value_to_r)].index        
        elif comparison_operator == "<":
            indexes = df[~(df[column] < value_to_r)].index        
        elif comparison_operator == "<=":
            indexes = df[~(df[column] <= value_to_r)].index        
        else:
            indexes = df[~(df[column] != value_to_r)].index
        
    df.drop(index=indexes, inplace=True)
    return df   

def get_Data(data, num_target):
    
    """
    Esta función devuelve las divisiones de datos finales para entrenar y probar

    Argumentos:
        - data: Dataframe elegido
        - num_target (int): target del dataframe

    Retornos:
        - Valor de retorno: Las divisiones de datos finales

    """
    y = data.iloc[:,num_target]
    X = data.drop(data.iloc[:,num_target], axis=1)
    return train_test_split(X, y, test_size = 0.33, shuffle = True, random_state = 45)

def borrar_html(texto):
    """
    Función para borrar la forma html

    Argumentos:
        - texto (str): Primer parametro.

    Retornos:
        - El texto sin forma html

    """
    html = re.compile('<.*?>')
    return html.sub('', texto)

def borrar_signos_puntuación(texto):
    
    """
    Función para borrar los signos de puntuación

    Argumentos:
        - texto (str): Primer parametro.

    Retornos:
        - El texto sin signos de puntuacion

    """
    tabla = str.maketrans('','',string.punctuation) 
    return texto.translate(tabla)

def borrar_url(texto):
    
    """
    Función para borrar los signos de las url

    Argumentos:
        - texto (str): Primer parametro.

    Retornos:
        - El texto sin los signos de las url

    """
    url = re.compile('https?://\S+|www\.\S+')
    return url.sub('',texto)

def encoder (df, column):

    """Función para hacer un labelEncoder.

        Argumentos:

        - df = Dataframe
        - column (int | str) = número o nombre de columna

    """
    
    if type(column) == int:
        encoder = LabelEncoder()
        df.iloc[:,column] = encoder.fit_transform(df.iloc[:,column])

    elif type(column) == str:
        encoder = LabelEncoder()
        df.loc[:,column] = encoder.fit_transform(df.loc[:,column])
    return df

def mean_Nan (df, column, parameter , new = False): 
        
    """Función para tratar los NaN.
    En función del parametro que introduzcamos nos rellenara los NaN con la media, con ceros, o con el valor especificado.

        Argumentos:
        - df = Dataframe
        - column (int/sring) = número o nombre de columna   
        - parameter (str) = 'mean', 'ceros,'fillna' 
        - new = valor nuevo que le quedamos dar a los Nan

    """
    
    if type(column) == int:
        if parameter == 'mean':
            df.iloc[:,column].fillna(value=df.iloc[:,column].mean() , inplace=True) 
        if parameter == 'ceros':
            df.iloc[:,column] = df.iloc[:,column].fillna(0)
        if parameter == 'fillna':
            df.iloc[:,column].fillna(new, inplace = True)

    elif type(column) == str:
        if parameter == 'mean':
            df.loc[:,column].fillna(value=df.loc[:,column].mean() , inplace=True) 
        if parameter == 'ceros':
            df.loc[:,column] = df.loc[:,column].fillna(0)
        if parameter == 'fillna':
            df.loc[:,column].fillna(new, inplace = True)
    return df

def renombrar_columna(df,val_ini,val_fin):
    '''Función para renombrar columnas:

    Argumentos:
    - df: Nombre del data frame
    - val_ini (str): Nombre de la columna
    - val_fin (str): Nombre nuevo que se le quiere dar a la columna

    '''

    df.rename(columns={val_ini:val_fin},inplace=True)
    return df

def limpiar(df,columna,palabra,nuevo_valor):
    '''Función Para sustituir una determinada palabra en una columna en concreto:
    
    Argumentos: 
    - df: Nombre del datframe
    - columna (str): Nombre de la columna
    - palabra (str|float|int): Palabara que se quiere sustituir
    - nuevo_valor (str|float|int): Nuevo valor que se le quiere dar a la palabra 

    '''

    df[columna] = df[columna].replace(palabra,nuevo_valor,regex=True)
    return df

def num_describe(data_in):
    """Retorna una mejor versión del describe

    Args:
        data_in (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: DataFrame con estadísticos
    """
    
    data_out = data_in.describe([.01,.02,.98,.99]).T
    data_out = data_out.drop(columns='count')
    data_out.insert(0,'skewness', data_in.skew())
    data_out.insert(0,'kurtosis', data_in.kurtosis())
    data_out.insert(0,'sparsity', (data_in==0).sum()/len(data_in))
    data_out.insert(0,'nulls', (data_in.isna()).sum()/len(data_in))
    return data_out

def read_images(path, size, color_space=None):
    '''Función para cargar las imágenes en un array. 

    Argumentos:
        path (str): path de la carpeta donde se encuentra la imagen o las imágenes
        size (tuple): (ancho de la imagen, altura de la imagen)
        filter (cv2.color_space): cambiar espacio de color de la imagen, por ejemplo cv2.COLOR_RGB2BGR (Default = None)

    Return:
        X : array
    '''

    X = []
    for img in os.listdir(path):
        image = cv2.imread(path+"/"+img)
        if color_space !=None:
            image = cv2.cvtColor(image, color_space)
        smallimage = cv2.resize(image, size)
        X.append(smallimage)
    return np.array(X)

def circ_distance(value_1, value_2, low, high):
    """Devuelve las distancias entre dos valores cíclicos.

    Argumentos:
        value_1 (int,float): primer valor
        value_2 (int,float): segundo valor
        low (int,float): descripcion
        high (int,float): descripcion

    Return:
        float: distancia entre dos valores
    """
    
    value_1_rad = ((value_1-low)/(high-low))*(2*np.pi)
    value_2_rad = ((value_2-low)/(high-low))*(2*np.pi)
    
    sin_value_1, cos_value_1 = np.sin(value_1_rad), np.cos(value_1_rad)
    sin_value_2, cos_value_2 = np.sin(value_2_rad), np.cos(value_2_rad)
    
    angle = np.arccos(np.dot([cos_value_1, sin_value_1],[cos_value_2, sin_value_2]))
    
    angle = angle*(high-low)/(2*np.pi) + low
    
    return round(angle,3)

def mice_impute_nans(df):
    ''' Función que utiliza la Imputación Múltiple por Ecuaciones Encadenadas.
        MICE realiza regresiones múltiples en muestras aleatorias de los datos y agregados para imputar los valores NaN.
    
        Argumentos:
        dataFrame = datos sin valores que faltan 
        
        Retornos:
        pandas.DataFrame: DataFrame con los missing values imputados
    ''' 

    cols = df.columns
    imputed_df = MICE( min_value= 0).fit_transform(df.values)
    imputed_df = pd.DataFrame(imputed_df)
    imputed_df.set_index(df.index, inplace=True)
    imputed_df.columns= cols

    return imputed_df

def convertidor_español(Dataframe, column_name):
    """ Conversión de números en formato string a float cuando estos tienen la puntuación 
    no anglosajona.
    Parámetros:
        - Dataframe: dataframe
        - Column_name (str): nombre de la columna
    Return : Dataframe   
        """

    Dataframe[column_name] = Dataframe[column_name].str.replace(".","")
    Dataframe[column_name] = Dataframe[column_name].str.replace(",",".").astype(float)

    return Dataframe

def convertidor_ingles(Dataframe, column_name):
    """ Conversión de números en formato string a float cuando estos tienen la puntuación 
    anglosajona.
    Argumentos:
        - Dataframe: dataframe
        - Column_name(str): nombre de la columna

    Return : Dataframe
        """

    Dataframe[column_name] = Dataframe[column_name].str.replace(",","").astype(float)
    return Dataframe

def normalize(palabra):
    """ Normalización de la palabras quitando símbolos de acentuación

    Argumentos:
        - palabra ( str): palabra que quereamos cambiar
        
    Return : palabra    
        """


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

    Argumentos: 
    Dataframe: Df
    Column_precio: str
    Nombre_columna: Str
    n: int.

    Return : Dataframe
    """

    lista_precios_n_periodos_atras = []

    for x,i in enumerate (Dataframe[Column_precio]):
        if x + n >= len(Dataframe[Column_precio]):
            lista_precios_n_periodos_atras.append(np.nan)
        else: 
            lista_precios_n_periodos_atras.append(Dataframe[Column_precio][x+1])
    Dataframe['PRECIO ANTERIOR']=lista_precios_n_periodos_atras
    Dataframe[nombre_nueva_columna]=(Dataframe[Column_precio]-Dataframe['PRECIO ANTERIOR'])/Dataframe['PRECIO ANTERIOR']

    return Dataframe

def beautifull_scrap(url, headers):
    """Función que obtiene la información de una página web mediante web scrapping.

        Argumentos:
        url (str): Dirección URL de la web que se quiere atacar.
        headers (dict): Diccionario con la cabecera que requiere la consulta a la web.

        Retornos:
        soup: HTML parseado de la librería bs4

    """

    response = requests.get(url, headers=headers)
    return bs(response.content, "lxml")

def suma (dicc, info):
    ''' Función para crear una nueva columna con la suma de los valores de columnas que le pases

    Argumentos:
    - dicc (dicc): LA key del diccionario sera la nueva columna y los values las columnas que nos van a sumar

    - info (dataframe): dataframe

    Retornos: 

    - Datarame con la columna con la ponderación
    '''

    for a in dicc.keys():
        info[a] = info[dicc[a]].sum(axis=1)
    return info

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