from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="omarpy",
    version="0.0.8",
    long_description = long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={'':'src'},
    packages=find_packages(where="src"),
    install_requires =[
        'pandas',
        'numpy',
        'regex',
        'scikit-learn',
        'opencv-python',
        'fancyimpute',
        'beautifulsoup4',
        'requests',
        'xgboost',
        'tensorflow',
        'prettytable',
        'Pillow',
        'matplotlib',
        'seaborn',
        'optuna',
        'nltk',
        'plotly',
        'wordcloud',
        'dash']
)