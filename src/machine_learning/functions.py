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