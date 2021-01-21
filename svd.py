# Data manipulation
import pandas as pd
import numpy as np

# Misc
import json

# Surprise package
import surprise
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from surprise import SVD

# for reproducible results
np.random.seed(3101)


def svd_test(df, cols=['UserId','ProductId','Score']):
    
    """to attain performance measure of SVD algo on the df input via cross validation
    
    Parameters:
    df (dataframe): data to be trained and tested via train_test_split for SVD recommendation
    cols (list): selected columns of the df (userid, productid, rating) for surprise package
    
    Returns:
    results verbose
    all_results (list): [test_rmse, fit_time, test_time], all 3 values are mean over a number of folds, here cv=5
    """
    
    df = df.drop_duplicates(subset=df.columns[2:].to_list(), keep='first') # df contains duplicate entries    
    df = df[cols].reset_index(drop=True)
    
    '''======== Purge infrequent consumers ========'''
    usercounts = df['UserId'].value_counts()
    user_filter = usercounts[usercounts > 2] # filter for users with more than 2 ratings
    user_filter = user_filter.index.tolist()
    df = df.loc[df['UserId'].isin(user_filter)]
    
    '''======== Purge infrequent stores ========'''
    pdtcounts = df['ProductId'].value_counts()
    product_filter = pdtcounts[pdtcounts > 9] # filter for products with more than 9 ratings
    product_filter = product_filter.index.tolist()
    df = df.loc[df['ProductId'].isin(product_filter)]
    
    max_qty = df['Score'].max()
    min_qty = df['Score'].min()
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df, reader)
    trainset = data.build_full_trainset()
    print('Number of users: ', trainset.n_users)
    print('Number of items: ', trainset.n_items)
    
    np.random.seed(3101)
    algo = SVD()
    algo.fit(trainset)
    results = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True) # RMSE as performance metric
    
    all_results = []
    for key in results:
        if type(results[key] == tuple):
            all_results.append(sum(results[key])/5)
        else:
            all_results.append(results[key].mean())
        
    return all_results   


def convert_rating(x, old_min, old_max, new_min, new_max):
    
    """converts quantity/rating to rating within a specified scale eg. 0 (new_min) to 5 (new_max)
    
    Parameters:
    x (int): quantity/rating to be converted to rating/new rating
    old_min (int): current minimum quantity, amongst all the users
    old_max (int): current maximum quantity, amongst all the users
    new_min (int): minimum rating that can be output
    new_max (int): maximum rating that can be output
    
    Returns:
    rating (int): quantity in the new specified range
    """
    
    diff = old_max - old_min
    new_diff = new_max - new_min
    rating = ((new_diff/diff)*(x-old_min))+new_min
    
    return rating
 
    
def svd_recommendation(fitted_model, df, user_id):
    
    """recommends products that user has not rated before, based on all users, with the SVD algorithm
    
    Parameters:
    fitted_model: train set already fitted with the algo i.e. algo.fit(trainset)
    df (dataframe): all data that is used for fitted_model, and to output SVD recommendation
    user_id (str): user to predict product recommendations for
    
    Returns:
    sort_recommendations (dict): sorted (descending) dictionary of ProductId as key and Est rating as value for recommended products
    """
    
    all_products = df['ProductId'].unique() # all unique products in the df given
    
    user_df = df[df['UserId'] == user_id]
    user_products = user_df['ProductId'].unique() # all unique products user_id has rated   
    unrated_products = np.setdiff1d(all_products, user_products) # unique products user_id has never rated before
    
    # data to predict on will comprise only of products user_id has not rated before 
    arbitrary_rating = 4.0 # since there are no ratings for products that user_id has never rated before
    new_data = [(user_id, pdt, arbitrary_rating) for pdt in unrated_products]    
    predictions = fitted_model.test(new_data)
    predictions_df = pd.DataFrame(predictions, columns=['UserId', 'ProductId', 'Score', 'Est', 'Details'])
    predictions_df.sort_values(by='Est', ascending=False, inplace=True)
       
    # product recommendations in descending order of predicted ratings
    recommendations = predictions_df[['ProductId', 'Est']].reset_index(drop=True)
    recommendations = recommendations.set_index('ProductId').to_dict().get('Est') # set ProductId as key, Est as value
    sort_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True) # sort products according to rating

    return dict(sort_recommendations)


def recommend(df, user_id, new_min, new_max):
    
    """recommends user_id products they have never rated before, according to what the users have rated previously
    
    Parameters:
    df (dataframe): data to be trained and tested on for SVD recommendation
    user_id (str): user identifier
    new_min and new_max will only change if range of ratings is to be changed,
    new_min (int): new minimum rating 
    new_max (int): new maximum rating
    
    Returns:
    recommended products (dict): dictionary with ProductId as key and predicted rating as value for products never rated by user
    dict is arranged in descending order of predicted ratings
    """
    
    df = df.drop_duplicates(subset=df.columns[2:].to_list(), keep='first') # df contains duplicate entries
    df = df[['UserId', 'ProductId', 'Score']].reset_index(drop=True)
    user_df = df[df['UserId'] == user_id]
    
    print("Purging infrequent customers...")
    '''======== Purge infrequent consumers ========'''
    usercounts = df['UserId'].value_counts()
    user_filter = usercounts[usercounts > 2] # filter for users with more than 2 ratings
    user_filter = user_filter.index.tolist()
    df = df.loc[df['UserId'].isin(user_filter)]
    
    print('Purging infrequent stores...')
    '''======== Purge infrequent stores ========'''
    pdtcounts = df['ProductId'].value_counts()
    product_filter = pdtcounts[pdtcounts > 9] # filter for products with more than 9 ratings
    product_filter = product_filter.index.tolist()
    df = df.loc[df['ProductId'].isin(product_filter)]
    
    max_qty = df['Score'].max()
    min_qty = df['Score'].min()
    
    # convert ratings to desired range, if needed
    if (max_qty != new_max) | (min_qty != new_min):
        df['Score'] = df['Score'].apply(convert_rating, args=(min_qty,max_qty,new_min,new_max)) 
        user_df = df[df['UserId'] == user_id]

    # load surprise package and convert data
    reader = Reader(rating_scale=(new_min, new_max)) # specify min and max ratings for rating_scale argument
    data = Dataset.load_from_df(df, reader)
    trainset = data.build_full_trainset()

    # instantiate SVD algo and fit the training dataset
    algo = SVD()
    model = algo.fit(trainset)

    recommended_products = svd_recommendation(model, df, user_id) # recommended products never rated before
    print(recommended_products)
    with open('output.json', 'w') as json_file:
        json.dump(recommended_products, json_file)

    return 'File saved: '+'output.json'
