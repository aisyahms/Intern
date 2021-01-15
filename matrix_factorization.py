# Data manipulation
import pandas as pd
import numpy as np

# Misc
# import pam_db
# import json 

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
    all_results (list): [test_rmse, fit_time, test_time], all 3 values are mean over a number of folds, here cv=5
    """
    
    df = df.drop_duplicates(subset=df.columns[2:].to_list(), keep='first') # df contains duplicate entries
    df = df[cols].reset_index(drop=True)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df, reader)
    trainset = data.build_full_trainset()
    print('Number of users: ', trainset.n_users, '\n')
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
    
    """recommends up to 10 top products that user has not rated before, based on other users, with the SVD algorithm
    
    Parameters:
    fitted_model: train set already fitted with the algo i.e. algo.fit(trainset)
    df (dataframe): all data that is used for fitted_model, and to output SVD recommendation
    user_id (str): user to predict product recommendations for
    
    Returns:
    recommendations (dataframe): includes ProductId for up to 10 top recommended products, specific for the user
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
    
    user_df_high = user_df[user_df['Score'] >= 4] # considering a rating of 4 or 5 as high
    if len(user_df_high) > 0:
        print(user_id + ' has rated items like this highly:')
        user_products_high = user_df_high['ProductId'].unique()
        print(user_products_high.tolist()[:5]) # output at most 5 products that have been rated highly by user_id
       
    # filter for top 10 recommendations for user
    print()
    print('Recommendations for ' + user_id + ':')
    recommendations = predictions_df[['ProductId']][:10]

    return recommendations


def recommend(df, user_id, new_min, new_max):
    
    """recommends user_id products according to what they have rated previously, 
    inclusive of (1) products they have never rated before, (2) products they have rated highly before
    
    Parameters:
    df (dataframe): data to be trained and tested on for SVD recommendation
    user_id (str)
    new_min and new_max will only change if range of ratings is to be changed,
    new_min (int): new minimum rating 
    new_max (int): new maximum rating
    
    Returns:
    svd_recommendation() output (list): list of up to 10 recommended ProductIds for user_id
    other_recommendations (list): list of up to 5 recommended ProductIds for the user_id,
    from previously highly rated purchased products, if any
    """
    
    df = df.drop_duplicates(subset=df.columns[2:].to_list(), keep='first') # df contains duplicate entries
    df = df[['UserId', 'ProductId', 'Score']].reset_index(drop=True)
    user_df = df[df['UserId'] == user_id]
    print("----User Rating for Products Dataframe----")
    print(user_df)
    print("------------------------------------------")

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

    recommended_products = svd_recommendation(model, df, user_id)
    print(recommended_products) # output up to 10 recommended ProductId

    # svd_recommendation doesn't include products that have been rated by user
    # we shall include products that have been highly rated by user to final recommendation results
    
    user_df = user_df.sort_values(by='Score', ascending=False)
    user_df = user_df.drop_duplicates(subset=user_df.columns[:-1].to_list(), keep='first')
    not_svd_rec = user_df['ProductId'][:5] # top 5 rated products by user
    
    other_recommendations = []
    for pdt in not_svd_rec:
        temp = user_df[user_df['ProductId'] == pdt].iloc[0] # only the first instance of pdt in user_df i.e. the highest rating
        if temp['Score'] >= 4: # considering a rating of 4 or 5 as high/ can repeat purchase      
            other_recommendations.append(pdt)
    
    print()
    if len(other_recommendations) != 0: # at least 1 product purchased before was rated highly by user
        print(user_id + ' can consider repeat purchases for products like:')
    else: # user has never rated any product purchased before highly
        other_recommendations = user_id + ' had no prior highly rated purchases.'

    return other_recommendations
