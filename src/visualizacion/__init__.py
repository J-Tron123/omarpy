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


from .visualizacion import (
    graf_bar_horizon,
    move_spines,
    plot_pairplot,
    stacked_bar_plot,
    y,
    h,
    num_plot,
    bar_hor,
    plot_bar_chart_with_numbers_y, 
    plot_boxplot, 
    data_report,
    grafico,
    scat_log_visualize,
    heatmap,
    graphs_sub,
    sweet_heatmap, 
    sweet_pie,
    sweet_cloud,
    plot_numerical_huetarget,
)
