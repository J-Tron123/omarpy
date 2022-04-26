from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import cv2
import os
import numpy as np

def remover_vello(imagen):
    '''
    Se aplica una máscara para eliminar el vello en la imagen. La función devuelve una `imagen` con las
    mismas dimensiones que la original.
    
    Parameters
    ----------
    imagen : img

    Output
    ------
    new_img : img

    '''
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

def mask_fondo(imagen):
    '''
    Función para eliminar el fondo de la imagen.

    Parameters
    ----------
    imagen : img

    Output
    ------
    new_img : img
    '''
    gray_example = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_example, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask) 
    new_img = cv2.bitwise_and(imagen,imagen,mask = mask_inv)
    return new_img

<<<<<<< HEAD

# Importacion de librerias usadas
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# list_maxComponents = [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
# list_maxDepth = [ 2, 3, 5, 10, 20, 30, 40]
# #list_maxFeatures = [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
# list_maxFeatures = [ (1/17), (2/17), (3/17), (4/17), (5/17), (6/17), (7/17), (8/17), (9/17), (10/17), (11/17), (12/17), (13/17), (14/17), (15/17), (16/17), (17/17)]

'''
    FUNCION: Mejor_PCA_DecissionTree_Regression
    FECHA: 25-04-2021
    VERSION: v0
    
    Funcion a la que se le pasa los datos de train, test, tipo de metrica, asi como los parametros del PCA y
    DecisionTreeRegressor, y devuelve los valores optimos de predicion, metrica y mejores parametros.
    
    Los parametros pasados son en el caso del:
        PCA: se pasa una lista con el n_components
        DecisionTreeRegressor: se pasa una lista con el max_depth, y otra lista con el max_features

    Argumentos:
        X_train (DataFrame): Cointains the independents vars splitted for training
        X_test (DataFrame):  Cointains the independents vars splitted for test
        y_train (DataFrame): Cointains the dependents vars splitted for training
        y_test (DataFrame): Cointains the dependents vars splitted for test
        metric (str): nombre de la metrica a utilizar en la prediccion. 
            Valores posibles ['mae','mape','mse','r2_score']
        list_maxComponents (list): lista con el n_components del PCA
        list_max_depth (list): lista con el max_depth del DecisionTreeRegressor
        list_maxFeatures (list): lista con el max_features del DecisionTreeRegressor
        Ejemplos:
            list_maxComponents = [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
            list_maxDepth = [ 2, 3, 5, 10, 20, 30, 40]
            list_maxFeatures = [ (1/17), (2/17), (3/17), (4/17), (5/17), (6/17), (7/17), (8/17), (9/17), (10/17), (11/17), (12/17), (13/17), (14/17), (15/17), (16/17), (17/17)]

    Retornos:
        y_pred (array): los valores optimos de la predicción
        metric_Best (float): el optimo valor de la metrica para los parametros dados
        n_components (int) el parametro optimo n_components del PCA
        max_depth (int): lista con el max_depth del DecisionTreeRegressor
        maxFeatures (float): lista con el maxFeatures del DecisionTreeRegressor

'''
def Mejor_PCA_DecissionTree_Regression(X_train, X_test, y_train, y_test, metric, 
                                        list_maxComponents, list_maxDepth, list_maxFeatures):

    # Escala los datos de train y test
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)

    
    n = 0 # Para indicar si es la primera vez
    for componentsi in list_maxComponents:
        pca = PCA(n_components=componentsi, random_state=42)
        pca.fit(X_train_scaled)
        X_train_scaled_pca = pca.transform(X_train_scaled)
        X_test_scaled_pca = pca.transform(X_test_scaled)
        for depthi in list_maxDepth:
            for featuresi in list_maxFeatures:
                dtr = DecisionTreeRegressor(max_depth = depthi, max_features = featuresi, random_state=42)
                dtr.fit(X_train_scaled_pca, y_train)
                y_pred = dtr.predict(X_test_scaled_pca)
                if (n == 0): # La primera vez se inicializa
                    if(metric == 'mae'):
                        metric_Best = mean_absolute_error(y_test, y_pred)
                    elif(metric == 'mape'):
                        metric_Best = mean_absolute_percentage_error(y_test, y_pred)
                    elif(metric == 'mse'):
                        metric_Best = mean_squared_error(y_test, y_pred)
                    elif(metric == 'r2_score'):
                        metric_Best = r2_score(y_test, y_pred)
                    else:
                        sys.exit('metric debe ser una de [\'mae\',\'mape\',\'mse\',\'r2_score\']')
                    max_components_Best = componentsi
                    max_depth_Best = depthi
                    max_features_Best = featuresi
                else:
                    if(metric == 'mae'):
                        metric_New = mean_absolute_error(y_test, y_pred)
                    elif(metric == 'mape'):
                        metric_New = mean_absolute_percentage_error(y_test, y_pred)
                    elif(metric == 'mse'):
                        metric_New = mean_squared_error(y_test, y_pred)
                    elif(metric == 'r2_score'):
                        metric_New = r2_score(y_test, y_pred)
                    else:
                        sys.exit('metric debe ser una de [\'mae\',\'mape\',\'mse\',\'r2_score\']')

                    if (metric == 'mae') or (metric == 'mape') or (metric == 'mse'): # Minimize
                        if (metric_New < metric_Best):
                            metric_Best = metric_New
                            max_components_Best = componentsi
                            max_depth_Best = depthi
                            max_features_Best = featuresi
                    if (metric == 'r2_score'): # Maximize
                        if (metric_Best < metric_New):
                            metric_Best = metric_New
                            max_components_Best = componentsi
                            max_depth_Best = depthi
                            max_features_Best = featuresi
            n +=1
    return y_pred, metric_Best, max_components_Best, max_depth_Best, max_features_Best

# Importacion de librerias usadas
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# list_maxComponents = [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
# list_n_estimators = [10, 25, 50, 100, 200, 300, 500, 800, 1000]
# list_max_leaf_nodes = [5, 10, 15, 20, 25]

'''
    FUNCION: Mejor_PCA_RandomForest_Regression
    FECHA: 25-04-2021
    VERSION: v0

    Funcion a la que se le pasa los datos de train, test, tipo de metrica, asi como los parametros del PCA y
    RandomForestRegressor, y devuelve los valores optimos de predicion, metrica y mejores parametros
    
    Los parametros pasados son en el caso del:
        PCA: se pasa una lista con el n_components
        RandomForestRegressor: se pasa una lista con el n_estimators, y una lista con el max_leaf_nodes 
    Argumentos:
        X_train (DataFrame): Cointains the independents vars splitted for training
        X_test (DataFrame):  Cointains the independents vars splitted for test
        y_train (DataFrame): Cointains the dependents vars splitted for training
        y_test (DataFrame): Cointains the dependents vars splitted for test
        metric (str): nombre de la metrica a utilizar en la prediccion. 
            Valores posibles ['mae','mape','mse','r2_score']
        list_maxComponents (list): lista con el n_components del PCA
        list_n_estimators (list): lista con el n_estimators del RandomForestRegressor
        list_max_leaf_nodes (list): lista con el max_leaf_nodes del RandomForestRegressor
        Ejemplos:
            list_maxComponents = [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
            list_n_estimators = [10, 25, 50, 100, 200, 300, 500, 800, 1000]
            list_max_leaf_nodes = [5, 10, 15, 20, 25]
   
    Retornos:
        y_pred (array): los valores optimos de la predicción
        metric_Best (float): el optimo valor de la metrica para los parametros dados
        n_components (int): el parametro optimo n_components del PCA
        n_estimators (int): el parametro optimo n_estimators del RandomForestRegressor
        max_leaf_nodes (int): el parametro optimo max_leaf_nodes del RandomForestRegressor

'''
def Mejor_PCA_RandomForest_Regression(X_train, X_test, y_train, y_test, metric, 
                                    list_maxComponents, list_n_estimators, list_max_leaf_nodes):
    # Escala los datos de train y test
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)

    n = 0 # Para indicar si es la primera vez
    for componentsi in list_maxComponents:
        pca = PCA(n_components=componentsi, random_state=42)
        pca.fit(X_train_scaled)
        X_train_scaled_pca = pca.transform(X_train_scaled)
        X_test_scaled_pca = pca.transform(X_test_scaled)
        for n_estimatorsi in list_n_estimators:
            for max_leaf_nodesi in list_max_leaf_nodes:
                rnd_reg = RandomForestRegressor(n_estimators=n_estimatorsi, max_leaf_nodes=max_leaf_nodesi, random_state=42)
                rnd_reg.fit(X_train_scaled_pca, y_train)
                y_pred = rnd_reg.predict(X_test_scaled_pca)
                if (n == 0): # La primera vez se inicializa
                    if(metric == 'mae'):
                        metric_Best = mean_absolute_error(y_test, y_pred)
                    elif(metric == 'mape'):
                        metric_Best = mean_absolute_percentage_error(y_test, y_pred)
                    elif(metric == 'mse'):
                        metric_Best = mean_squared_error(y_test, y_pred)
                    elif(metric == 'r2_score'):
                        metric_Best = r2_score(y_test, y_pred)
                    else:
                        sys.exit('metric debe ser una de [\'mae\',\'mape\',\'mse\',\'r2_score\']')
                    n_estimators_Best = n_estimatorsi
                    max_leaf_nodes_Best = max_leaf_nodesi
                    max_components_Best = componentsi
                else:
                    if(metric == 'mae'):
                        metric_New = mean_absolute_error(y_test, y_pred)
                    elif(metric == 'mape'):
                        metric_New = mean_absolute_percentage_error(y_test, y_pred)
                    elif(metric == 'mse'):
                        metric_New = mean_squared_error(y_test, y_pred)
                    elif(metric == 'r2_score'):
                        metric_New = r2_score(y_test, y_pred)
                    else:
                        sys.exit('metric debe ser una de [\'mae\',\'mape\',\'mse\',\'r2_score\']')

                    if (metric == 'mae') or (metric == 'mape') or (metric == 'mse'): # Minimize
                        if (metric_New < metric_Best):
                            metric_Best = metric_New
                            n_estimators_Best = n_estimatorsi
                            max_leaf_nodes_Best = max_leaf_nodesi
                            max_components_Best = componentsi
                    if (metric == 'r2_score'): # Maximize
                        if (metric_Best < metric_New):
                            metric_Best = metric_New
                            n_estimators_Best = n_estimatorsi
                            max_leaf_nodes_Best = max_leaf_nodesi
                            max_components_Best = componentsi
            n +=1
    return y_pred, metric_Best, max_components_Best, n_estimators_Best, max_leaf_nodes_Best

# Importacion de librerias usadas
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# list_maxComponents = [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
# list_n_estimators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
# list_max_depth = [5, 10, 15, 20, 25]
# list_learning_rate = [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5]

'''
    FUNCION: Mejor_PCA_XGB_Regression
    FECHA: 25-04-2021
    VERSION: v0

    Funcion a la que se le pasa los datos de train, test, tipo de metrica, asi como los parametros del PCA y
    XGBRegressor, y devuelve los valores optimos de predicion, metrica y mejores parametros.
    
    Los parametros pasados son en el caso del:
        PCA: se pasa una lista con el n_components
        XGBRegressor: se pasa una lista con el n_estimators, una lista con el max_depth, y 
            otra lista con el learning_rate
    Argumentos:
        X_train (DataFrame): Cointains the independents vars splitted for training
        X_test (DataFrame):  Cointains the independents vars splitted for test
        y_train (DataFrame): Cointains the dependents vars splitted for training
        y_test (DataFrame): Cointains the dependents vars splitted for test
        metric (str): nombre de la metrica a utilizar en la prediccion. 
            Valores posibles ['mae','mape','mse','r2_score']
        list_maxComponents (list): lista con el n_components del PCA
        list_n_estimators (list): lista con el n_estimators del XGBRegressor
        list_max_depth (list): lista con el max_depth del XGBRegressor
        list_learning_rate (list): lista con el learning_rate del XGBRegressor
        Ejemplos:
            list_maxComponents = [ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
            list_n_estimators = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            list_max_depth = [5, 10, 15, 20, 25]
            list_learning_rate = [0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5]
    Retornos:
        y_pred (array): los valores optimos de la predicción
        metric_Best (float): el optimo valor de la metrica para los parametros dados
        n_components (int) el parametro optimo n_components del PCA
        n_estimators (int): el parametro optimo n_estimators del XGBRegressor
        max_depth (int): lista con el max_depth del XGBRegressor
        learning_rate (float): lista con el learning_rate del XGBRegressor

'''
def Mejor_PCA_XGB_Regression(X_train, X_test, y_train, y_test, metric, 
                            list_maxComponents, list_n_estimators, list_max_depth, list_learning_rate):
    # Escala los datos de train y test
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)

    n = 0 # Para indicar si es la primera vez
    for componentsi in list_maxComponents:
        pca = PCA(n_components=componentsi, random_state=42)
        pca.fit(X_train_scaled)
        X_train_scaled_pca = pca.transform(X_train_scaled)
        X_test_scaled_pca = pca.transform(X_test_scaled)
        for n_estimatorsi in list_n_estimators:
            for max_depthi in list_max_depth:
                for learning_ratei in list_learning_rate:
                    xgb_reg = XGBRegressor(n_estimators=n_estimatorsi, max_depth=max_depthi, learning_rate=learning_ratei, random_state=42)
                    xgb_reg.fit(X_train_scaled_pca, y_train)
                    y_pred = xgb_reg.predict(X_test_scaled_pca)
                    if (n == 0): # La primera vez se inicializa
                        if(metric == 'mae'):
                            metric_Best = mean_absolute_error(y_test, y_pred)
                        elif(metric == 'mape'):
                            metric_Best = mean_absolute_percentage_error(y_test, y_pred)
                        elif(metric == 'mse'):
                            metric_Best = mean_squared_error(y_test, y_pred)
                        elif(metric == 'r2_score'):
                            metric_Best = r2_score(y_test, y_pred)
                        else:
                            sys.exit('metric debe ser una de [\'mae\',\'mape\',\'mse\',\'r2_score\']')
                        n_estimators_Best = n_estimatorsi
                        max_depth_Best = max_depthi
                        learning_rate_Best = learning_ratei
                        max_components_Best = componentsi
                    else:
                        if(metric == 'mae'):
                            metric_New = mean_absolute_error(y_test, y_pred)
                        elif(metric == 'mape'):
                            metric_New = mean_absolute_percentage_error(y_test, y_pred)
                        elif(metric == 'mse'):
                            metric_New = mean_squared_error(y_test, y_pred)
                        elif(metric == 'r2_score'):
                            metric_New = r2_score(y_test, y_pred)
                        else:
                            sys.exit('metric debe ser una de [\'mae\',\'mape\',\'mse\',\'r2_score\']')

                        if (metric == 'mae') or (metric == 'mape') or (metric == 'mse'): # Minimize
                            if (metric_New < metric_Best):
                                metric_Best = metric_New
                                n_estimators_Best = n_estimatorsi
                                max_depth_Best = max_depthi
                                learning_rate_Best = learning_ratei
                                max_components_Best = componentsi
                        if (metric == 'r2_score'): # Maximize
                            if (metric_Best < metric_New):
                                metric_Best = metric_New
                                n_estimators_Best = n_estimatorsi
                                max_depth_Best = max_depthi
                                learning_rate_Best = learning_ratei
                                max_components_Best = componentsi
            n +=1
    return y_pred, metric_Best, max_components_Best, n_estimators_Best, max_depth_Best, learning_rate_Best
=======
def scaler(scaler: str, data: np.array):
    """
    Scales the data.

    Args:
        scaler: A scaler to choose from sklearn library. StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
        data: Array like data to be scaled.
    
    Returns:
        The data scaled.
    """
    try:
        if scaler=='StandardScaler':
            return StandardScaler().fit_transform(data)
        if scaler=='MinMaxScaler':
            return MinMaxScaler().fit_transform(data)
        if scaler=='MaxAbsScaler':
            return MaxAbsScaler().fit_transform(data)
        if scaler=='RobustScaler':
            return RobustScaler().fit_transform(data)
            
    except ValueError:
        print('Choose one of the scalers listed.')
    
def run_model(X_train, X_test, y_train, y_test, model_name, params): # params = funcion Miguel
    ''' Esta función sirve para correr los diferentes modelos de machine learning.

    Args:
        X_train, X_test, y_train, y_test: división del dataset en train y test.
        model_name: modelo que se quiere probar ***(ej. LogisticRegression, RandomForestClassifier, XGBoost...).***
        params: (***funciones de Miguel***) (ej.: función X - para x, función Y, para y.)
    
    Return:
        El modelo entrenado con los parámetros indicados en la llamada de la función.
    '''
    model = model_name(params)
    model.fit(X_train, y_train)
    return model

def prediction(model, X_test):
    ''' Función para relaizar las predicciones del modelo de machine learning sobre la parte de test.

    Args:
        model: indicar la variable correspondiente al modelo entrenado.
        X_test: indicar la variable correspondiente a test sobre la que se van a realizar las predicciones.
    
    Return:
       Un array con las predicciones realizadas.
    '''
    preds = model.predict(X_test).round(0)
    return preds

def c_mat(y_test, X_test, model):
    ''' Generación de una matriz de confusión a partir de los resultados 
    obtenidos de las predicciones realizadas sobre la parte de test.

    Args:
        y_test: variable que contiene la 'target' de la parte de test.
        X_test: variable que contiene las 'features' de la parte de test.

    Return:
        La matriz de confusión en base a los argumentos introducidos.
    '''
    from sklearn.metrics import confusion_matrix
    
    c_mat = confusion_matrix(y_test, model.predict(X_test).round(0))
    return c_mat

def class_results(y_test, pred_y):
    ''' Resultados obtenidos a partir de un modelo de classificación.

    Args:
        y_test: variable con la 'target' de test.
        pred_y: variable con las predicciones sobre el test.

    Return:
        Reporte sobre los resultados obtenidos.
        Matriz de confusión mostrada con un 'heatmap' de seaborn.
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(9,6))
    sns.heatmap(conf_matrix, annot=True)
    plt.title('Confusion matrix')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    print (classification_report(y_test, pred_y))

def binary_class_metrics(y_train, y_test):
    ''' Resultado de las métricas de accuracy, precision, recall y
    f1 score para modelos de clasificación binaria.

    Args:
        y_train: variable con la 'target' de la parte de train.
        y_test: variable con la 'target' de la parte de test.
    
    Return:
        Print de las 4 métricas indicadas: accuracy, precision,
        recall, f1 score.
    '''
    from sklearn import metrics

    accuracy = metrics.accuracy_score(y_train, y_test)
    print('Accuracy score:', accuracy)

    precision = metrics.precision_score(y_train, y_test)
    print('Precision score:', precision)

    recall = metrics.recall_score(y_train, y_test)
    print('Recall score:', recall)

    f1_score = metrics.f1_score(y_train, y_test)
    print('F1 score:', f1_score)

def precision_recall_AUC(y_train, y_test):
    ''' Resultado de la métrica AUC a partir del modelo
    entrenado.

    Args:
        y_train: variable con la 'target' de la parte de train.
        y_test: variable con la 'target' de la parte de test.
    
    Return:
        El score de AUC en base a los argumentos indicados.
    '''
    from sklearn import metrics

    recall = metrics.recall_score(y_train, y_test)
    precision = metrics.precision_score(y_train, y_test)
    auc = auc(recall, precision)

    return auc
>>>>>>> b11f85c06f148412bc3fda9f0da28b332ddcb597
