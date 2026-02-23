from setuptools import setup, find_packages

setup(
    name='FinanceNetworks',
    version='0.1.0',
    author='Aaryam Sharma',
    description='A project for ECON 423 Final Project on Finance Networks',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'yfinance',
        'scikit-learn',
        'arch',
        'tqdm',
        'statsmodels',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)