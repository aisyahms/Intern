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
    - The data set should ideally contain columns including User Id, Product Id, and Rating.
    - These 3 variables are needed to make up the data to pass through surprise package's functions.

- cols (list): list of column names to be used as filtered data passing through surprise's `Dataset` and `Reader` functions
    - For NewProfile.csv, `cols = ['UserId', 'ProductId', 'Score']`

`recommend(df, user_id, new_min, new_max)` function:

- df (dataframe): data set containing rows of product review/rating by users, same as df used in `svd_test`

- user_id (str): user identification number/name
    - The user/customer that the script's functions should output recommendations for, based on the data set input.

- new_min (int), new_max (int): current, or specified new range, for ratings by users
    - The minimum and maximum values of ratings should be taken across all entries and users. 
    - Range values will be used to pass through `Reader` function in surprise.


## How to Train
------------
To test the performance of the model using the `SVD()` algorithm, input the arguments: df and cols into `svd_test(df, cols)`, as specified above.

To train the model using the data input, input the arguments: df, user_id, new_min and new_max into `recommend(df, user_id, new_min, new_max)` as specified above.
In the `recommend` function, duplicates of entries will be dropped, and only the 3 main cols will be used as part of the data loaded and read into surprise.
```python
df = df.drop_duplicates(subset=df.columns[2:].to_list(), keep='first') # df contains duplicate entries
df = df[['UserId', 'ProductId', 'Score']].reset_index(drop=True)
```
```python
reader = Reader(rating_scale=(new_min, new_max)) # specify min and max ratings for rating_scale argument
data = Dataset.load_from_df(df, reader)
trainset = data.build_full_trainset()
```
The SVD algorithm is initiated and the data input is fitted as the training data. 
```python
algo = SVD()
model = algo.fit(trainset)
```
Using the intermediate `svd_recommendation` function, `df` is used to generate `new_data` by finding unique products that the user has not rated before. The SVD algorithm then predicts the user's ratings for these products.
```python
all_products = df['ProductId'].unique() # all unique products in the df given
    
user_df = df[df['UserId'] == user_id]
user_products = user_df['ProductId'].unique() # all unique products user_id has rated   
unrated_products = np.setdiff1d(all_products, user_products) # unique products user_id has never rated before
    
# data to predict on will comprise only of products user_id has not rated before 
arbitrary_rating = 4.0 # since there are no ratings for products that user_id has never rated before
new_data = [(user_id, pdt, arbitrary_rating) for pdt in unrated_products]    
predictions = fitted_model.test(new_data)
```


## Data Generated 
------------
By running the main function, `recommend`, 
1. a dataframe of products and their respective ratings by the user_id specified in the function argument <br />
If the user has products previously rated highly (defined as a rating of 4 or 5 here), <br />
2. a maximum of 5 of those products will be output as a reminder of what their good previous purchases look like <br />
3. a maximum of 10 recommendations of products they have never rated before, based on other users and items they have rated <br />
4. a maximum of 5 other product recommendations based on what that they have rated highly before (similar to 2.)
```
----User Rating for Products Dataframe----
                UserId   ProductId  Score
4       A1UQRSCLF8GW1T  B006K2ZZ7K      5
346686  A1UQRSCLF8GW1T  B003XDH6M6      5
------------------------------------------
A1UQRSCLF8GW1T has rated items like this highly:
['B006K2ZZ7K', 'B003XDH6M6']

Recommendations for A1UQRSCLF8GW1T:
        ProductId
12949  B000KEJMRI
8193   B000ES5GL6
7590   B000ED9L9E
14287  B000LRIFU4
25687  B001E5E3B2
25364  B001E53TMQ
7603   B000EDBQ6A
25700  B001E5E3X0
30901  B001LO37PY
23345  B0018CFN92

A1UQRSCLF8GW1T can consider repeat purchases for products like:
Out[1]: ['B006K2ZZ7K', 'B003XDH6M6']
```
If the user has never rated products highly (defined as a rating of 4 or 5), <br />
2. a maximum of 10 recommendations of products they have never rated before, based on other users and items they have rated <br />
3. string output mentioning that the user has no prior highly rated purchases
```
----User Rating for Products Dataframe----
           UserId   ProductId  Score
1  A1D87F6ZCVE5NK  B00813GRG4      1
------------------------------------------

Recommendations for A1D87F6ZCVE5NK:
        ProductId
35749  B0029O0XGQ
13656  B000LKXGKU
33771  B001VNGG58
7590   B000ED9L9E
9693   B000FK7PQW
29807  B001HXNIPS
8725   B000F3YEPO
25364  B001E53TMQ
53302  B004IS56Y0
9632   B000FICDO8

Out[2]: 'A1D87F6ZCVE5NK had no prior highly rated purchases.'
```
