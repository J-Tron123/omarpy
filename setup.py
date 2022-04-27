from setuptools import setup, find_packages

setup(
    name="omarpy",
    version="0.0.1",
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
    ]
)