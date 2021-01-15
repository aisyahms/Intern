# Introduction
------------
matrix_factorization.py is a Singular Value Decomposition (SVD) script meant as a Collaborative Filtering approach for Recommender Systems. The script utilises the [Surprise](http://surpriselib.com/) package to build a trainset using existing data, and to output recommendations for a particular user. 


# Requirements
------------
The module requires the following libraries:

1. [NumPy](https://numpy.org/)
2. [pandas](https://pandas.pydata.org/)
3. [surprise](http://surpriselib.com/)

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Surprise.
```bash
pip install scikit-surprise
```
For data manipulation, 
```python
import pandas as pd
import numpy as np
```
From the surprise package, 
```python
import surprise
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from surprise import SVD
```


# Data Required
------------






# How to Train
------------






# Data Generated 
------------





