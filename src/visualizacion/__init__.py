"""
El modulo :mod:`omarpy.visualizacion` incluye funciones que ayudan con la visualización de datos. 
"""

import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


from .functions import (
    graf_bar_horizon,
    f1,
    move_spines,
    plot_pairplot,
    stacked_bar_plot,
    num_plot,
    bar_hor,
    plot_bar_chart_with_numbers_y, 
    plot_boxplot, 
    data_report,
    grafico,
    scat_log_visualize,
    sweet_heatmap, 
    sweet_pie_1,
    sweet_pie2,
    sweet_cloud,
    plot_numerical_huetarget,
    escribir_a_color,
    plot_2dline, 
    top100plot
)
