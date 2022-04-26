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
