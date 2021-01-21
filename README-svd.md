# Introduction
------------
svd.py is a Singular Value Decomposition (SVD) script meant as a Collaborative Filtering approach for Recommender Systems. The script utilises the [surprise](http://surpriselib.com/) package, which deals with explicit rating data. The package is used to build a trainset using existing data, and to return rating predictions for a particular user where products with rating predictions on the higher spectrum can be output as recommendations specific for the user. 

# Data Required
------------
For `svd_test`, the function requires the following:

Argument | Data Type | Example | Description |
--- | --- | --- | --- |
df | dataframe | example below | data set containing rows of product review/rating by users |
cols | list | ['UserId', 'ProductId', 'Score'] | list of column names to filter data passing through surprise's `Dataset` and `Reader` functions |

df example:

Field | Data Type | Example | Description | 
--- | --- | --- | --- |
UserId | string | A3SGXH7AUHU8GW | The identifier of the user |
ProductId | string | B001E4KFG0 | The identifier of the product |
Score | integer | 5 | The rating given by the user for the product | 


For `recommend`, the function requires the following:

Argument | Data Type | Example | Description |
--- | --- | --- | --- |
df | dataframe | example above | data set containing rows of product review/rating by users, same df used in `svd_test` |
user_id | string | A3SGXH7AUHU8GW | identifier of the user that the function will output recommendations for |
new_min | integer | 1 | current/specified new minimum value of ratings across all entries and users |
new_max | integer | 5 | current/specified new maximum value of ratings across all entries and users |

# How to Train
------------
To test the performance of the model using the `SVD()` algorithm, input the arguments: df and cols into `svd_test(df, cols)`, as specified above.

To train the model using the data input, input the arguments: df, user_id, new_min and new_max into `recommend(df, user_id, new_min, new_max)` as specified above.\
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

# Data Generated 
------------
By running the main function, `recommend`, it will return a JSON object with nested dictionaries. The first nested dictionary comprises of New Recommendations, recommendations of products the user has never rated before. The second nested dictionary comprises of Other Recommendations, recommendations of products where user has rated highly before (defined as a rating of 4 or 5) and can consider purchasing again (repeat purchase).

`
{'New Recommendations': {'B0000DK4ZR': 5.0,
  'B004SOXDFI': 5.0,
  'B0044MTGHI': 5.0,
  'B001EQ554K': 5.0,
  'B0001IOSDQ': 5.0,
  'B000FK7PQW': 5.0,
  'B000O5DI3W': 5.0,
  'B000O5DI1E': 5.0,
  'B000F3WSBQ': 5.0,
  'B0027XYIDI': 5.0,
  'B001E5E10A': 5.0,
  'B000E7VK9Y': 5.0,
  'B000E7SYKM': 5.0,
  'B001EO6EMY': 4.991322114829467,
  'B000PW05RQ': 4.990090413880482},
 'Other Recommendations': {'B004DMGQKE': 5, 'B000LQOCH0': 4, 'B000CQ26E0': 4}}
`

If the user has never rated products highly (defined as a rating of 4 or 5), Other Recommendations' dictionary will be empty

# Libraries used 
------------

1. [NumPy](https://numpy.org/)
2. [pandas](https://pandas.pydata.org/)
3. [surprise](http://surpriselib.com/)

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Surprise.
```bash
pip install scikit-surprise
```
