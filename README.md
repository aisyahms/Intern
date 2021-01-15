## Introduction
------------
matrix_factorization.py is a Singular Value Decomposition (SVD) script meant as a Collaborative Filtering approach for Recommender Systems. The script utilises the [surprise](http://surpriselib.com/) package, which deals with explicit rating data. The package is used to build a trainset using existing data, and to return rating predictions for a particular user where products with rating predictions on the higher spectrum can be output as recommendations specific for the user. 


## Requirements
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


## Data Required
------------
`svd_test(df, cols)` function:

- df (dataframe): data set containing rows of product review/rating by users
    1. The data set should ideally contain columns including User Id, Product Id, and Rating.
    2. These 3 variables are needed to make up the data to pass through surprise package's functions.

- cols (list): list of column names to be used as filtered data passing through surprise's `Dataset` and `Reader` functions
    1. For NewProfile.csv, `cols = ['UserId', 'ProductId', 'Score']`

`recommend(df, user_id, new_min, new_max)` function:

- df (dataframe): data set containing rows of product review/rating by users, same as df used in `svd_test`

- user_id (str): user identification number/name
    1. The user/customer that the script's functions should output recommendations for, based on the data set input.

- new_min (int), new_max (int): current, or specified new range, for ratings by users
    1. The minimum and maximum values of ratings should be taken across all entries and users. 
    2. Range values will be used to pass through `Reader` function in surprise.


## How to Train
------------
To train, input the arguments required (df, user_id, new_min, new_max), specified in Data Required above, into `recommend`.
In the `recommend` function, duplicates of entries will be dropped, and only the 3 main columns will be used as part of the dataset loaded into surprise.
The SVD algorithm is then initiated and the data set is fitted as the training data. 


## Data Generated 
------------
Output of products and ratings by user specified in input. 
If user has products previously rated highly, a maximum of 5 of those products will be output.
Up to 10 recommendations (never rated before)
Up to 5 of other recommendations  (rated before) will be output, should the user have rated these highly before. 

If user has never rated products highly before, 
Up to 10 recommendations (never rated before)
A string saying that the user has no prior highly rated purchases.

