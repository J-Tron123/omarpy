



'''FUNCION QUE PONDERA COLUMNAS LAURA'''


def ponderacion (dicc, info):
    ''' Funcion que pondera columnas en funcion de unas categorias  (keys) pasadas a traves de un diccionario cuyos valores son los nombres de las columnas
    de los que se van a tomar los valores para hacer la suma

    INPUT:
    dicc (dicc): diccionario cuyas keys hacen referencia a categorias que necesitamos establecer en un dataframe (a modo conceptual o practico) con el nombre de las columnas
    que hacen referencia a esa categoria
    info (dataframe): dataframe

    Autor Laura
    '''

    for a in dicc.keys():
       info[a] = info[dicc[a]].sum(axis=1)
    return info


'''FUNCION QUE CAMBIA VALORES EN COLUMNAS LAURA'''


def cambio_respuestas (info,dicc, inicio,fin):

    ''' Funcion que permite cambiar los valores de unas columnas dadas

    INPUT:
    info (dataframe): dataframe
    dicc (dicc): diccionario cuyas keys hacen referencia a los valores de una o varias columnas y cuyos valores son por los que se deben cambiar
    inicio (int): numero de columna desde donde hay que hacer los cambios
    fin (int): numero de columna (+1) hasta la que hay que hacer los cambios

    Autor Laura
    '''

    for a in range(info.iloc[ : ,inicio:fin].shape[1]):
        info.iloc[: ,a] = info.iloc[: ,a].map(dicc)
    return info