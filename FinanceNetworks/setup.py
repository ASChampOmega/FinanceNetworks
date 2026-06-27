from setuptools import setup, find_packages

setup(
    name='FinanceNetworks',
    version='0.2.0',
    author='Aaryam Sharma',
    description='Volatility forecasting with network-augmented econometric models',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.4.3',
        'pandas>=3.0.1',
        'matplotlib>=3.10.8',
        'scipy>=1.17.1',
        'yfinance>=1.2.0',
        'scikit-learn>=1.8.0',
        'arch>=8.0.0',
        'tqdm>=4.67.3',
        'statsmodels>=0.14.6',
        'networkx>=3.6.1',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)