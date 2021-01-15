# Introduction
------------
matrix_factorization.py is a Singular Value Decomposition (SVD) script meant as a Collaborative Filtering approach for Recommender Systems. The script utilises the [surprise](http://surpriselib.com/) package, which deals with explicit rating data. The package is used to build a trainset using existing data, and to return rating predictions for a particular user where products with rating predictions on the higher spectrum can be output as recommendations specific for the user. 


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
- df (dataframe): data set containing rows of item review/rating by users
df should ideally contain the rows: User Id, Product Id, and Rating 
These 3 variables are needed to make up the data to pass through surprise package's functions.

- user_id (str): user identification
The user/customer that the script functions should output recommendations for, based on the df input.

- range of ratings
minimum rating (int) and maximum rating (int) found across all users in the data set input.


# How to Train
------------
To train, input the arguments required (df, user_id, new_min, new_max), specified in Data Required above, into `recommend`.
In the `recommend` function, duplicates of entries will be dropped, and only the 3 main columns will be used as part of the dataset loaded into surprise.
The SVD algorithm is then initiated and the data set is fitted as the training data. 


# Data Generated 
------------
Output of products and ratings by user specified in input. 
If user has products previously rated highly, a maximum of 5 of those products will be output.
Up to 10 recommendations (never rated before)
Up to 5 of other recommendations  (rated before) will be output, should the user have rated these highly before. 

If user has never rated products highly before, 
Up to 10 recommendations (never rated before)
A string saying that the user has no prior highly rated purchases.

