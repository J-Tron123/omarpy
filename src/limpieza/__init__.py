"""El modulo :mod:omarpy.limpieza incluye funciones que ayudan con la limpieza de datos."""

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

from .functions import (
    drop_missings,
    remove_units, 
    to_type, 
    filter_df,
    col_to_float, 
    nan_as_nan, 
    inf_as_nan, 
    regex_tex_all, 
    regex_tex_first, 
    drop_rows_by_index, 
    get_Data, 
    borrar_html, 
    borrar_signos_puntuación, 
    borrar_url, 
    encoder, 
    mean_Nan, 
    renombrar_column, 
    limpiar, 
    num_describe, 
    read_images, 
    circ_distanc, 
    mice_impute_nans, 
    convertidor_español,
    convertidor_ingles,
    normalize, 
    crear_rentabilidades, 
    beautifull_scrap,
    suma, 
    contar_imagenes, 
    create_dict_images, 
    remover_vello, 
    mask_fondo
)
