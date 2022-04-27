"""
El modulo :mod:`omarpy.visualizacion` incluye funciones que ayudan con la visualización de datos. 
"""

import seaborn as sns
import pandas as pd
import numpy as np

from plotly.offline import iplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from .visualizacion import (
    data_report,
    grafico,
    scat_log_visualize,
    heatmap,
    graphs_sub
)
