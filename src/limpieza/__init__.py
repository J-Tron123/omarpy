"""El modulo :mod:omarpy.visualizacion incluye funciones que ayudan con la visualización de datos."""

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
    num_describe,
    read_images,
    circ_distance,
    inf_as_nan,
    mice_impute_nans,
    remove_units,
    convertidor_español,
    convertidor_ingles,
    normalize, 
    crear_rentabilidades, 
    columnascat,
    beautifull_scrap,
    drop_missings,
    file_sorter
)