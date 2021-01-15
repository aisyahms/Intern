import pandas as pd
import numpy as np
import datetime as dt
import os
import pam_db
import surprise
from surprise import SVD
# from surprise import NMF
# from surprise import accuracy
# from surprise.model_selection import train_test_split
import json



#Writing the SVD recommendation function
def svd_recommendation(algo, cust_ratings, customer_id):

    arbitrary_num = 4

    iids = cust_ratings['promotion_id'].unique()
    print("iids="+str(iids))
    #This first two lines of code are very important as we need to find out the promotionId screens which the specified user hasn't rated
    cust_rating_of_specific_cust = cust_ratings[cust_ratings['customer_id'] == customer_id]
    print("cust_rating_of_specific_cust=" + str(cust_rating_of_specific_cust))
    iids_unrated = np.setdiff1d(iids, cust_rating_of_specific_cust)
    print("iids_unrated=" + str(iids_unrated))
    new_data = [[customer_id, iid, arbitrary_num] for iid in iids_unrated]
    print("new_data=" + str(new_data))
    new_predictions = algo.test(new_data)
    print("new_predictions=" + str(new_predictions))

    #Collect the prediction ratings
    predicted_ratings = np.array([prediction.est for prediction in new_predictions])
    print("predicted_ratings=" + str(predicted_ratings))

    #This line of code will extract the index of the top 5 predicted ratings via argsort()
    list_of_index_in_order_of_highest_predicted_rating = predicted_ratings.argsort()[::-1]
    print("list_of_index_in_order_of_highest_predicted_rating=" + str(list_of_index_in_order_of_highest_predicted_rating))
    list1 = iids_unrated[list_of_index_in_order_of_highest_predicted_rating.tolist()].tolist()
    print("list1=" + str(list1))
    list2 = cust_rating_of_specific_cust["promotion_id"].tolist()
    print("list2=" + str(list2))
    list1.extend(list2)
    print("list1=" + str(list1))

    return list1


def quantity_to_rating(x, step_size, max_qty):
    #Extract the max and min ratings first and then taking difference between both.

    if (x <= max_qty) & (x > (max_qty - step_size)):
        return 5
    elif (x <= max_qty - step_size) & (x > max_qty - step_size * 2):
        return 4
    elif (x <= max_qty - step_size * 2) & (x > max_qty - step_size * 3):
        return 3
    elif (x <= max_qty - step_size * 3) & (x > max_qty - step_size * 4):
        return 2
    else:
        return 1


def recommend(customer_id):

    pam_db_mgr = pam_db.PamDbMgr()
    # interest_value_for_all_promotions = pam_db_mgr.get_interest_value_for_all_promotions(customer_id)
    interest_value_for_all_promotions = pam_db_mgr.get_interest_value_for_all_promotions()
    interest_value_for_all_promotions_df = pd.DataFrame(interest_value_for_all_promotions, columns=['customer_id', 'promotion_id', 'quantity'])

    print("----interest_value_for_all_promotions_df----")
    print(interest_value_for_all_promotions_df)
    print("--------------------------------------------")


    #Extract the max and min ratings first and then taking difference between both.
    max_qty = interest_value_for_all_promotions_df['quantity'].max()
    min_qty = interest_value_for_all_promotions_df['quantity'].min()
    print("max_qty="+str(max_qty))
    print("min_qty=" + str(min_qty))
    diff_qty = max_qty - min_qty
    print("diff_qty=" + str(diff_qty))
    print("algo 13")

    #Now I will need to write a function to convert the Quantity to ratings. This is based on the assumption that the more people visit a screen
    #the more likely they are to rate it highly. I took the difference between the max and min quantity divided by 5 because I wanted ti to be converted to
    #a 5 point rating scale basis
    step_size = diff_qty/5
    print("step_size=" + str(step_size))

    interest_value_for_all_promotions_df['quantity'] = interest_value_for_all_promotions_df['quantity'].apply(quantity_to_rating, args=(step_size, max_qty))

    interest_value_for_all_promotions_df.rename(columns={'quantity':'ratings'}, inplace=True)
    print("algo 15")


    #loading the surprise package and converting the data into "surprise" format. I would need to specify the max and min ratings
    reader = surprise.Reader(rating_scale=(interest_value_for_all_promotions_df['ratings'].min(),interest_value_for_all_promotions_df['ratings'].max()))
    data = surprise.Dataset.load_from_df(interest_value_for_all_promotions_df, reader)


    #Instantiating the SVD algo first and fitting the training dataset
    algo = SVD()
    algo.fit(data.build_full_trainset())

    recommended_prod_ids = svd_recommendation(algo, interest_value_for_all_promotions_df, customer_id)

    # the svd operation might not involves all promotions. We need to append promotion ids that are to involved to 
    # the result before output the result

    all_promotion_ids = pam_db_mgr.get_all_promotions_id()
    print("type(all_promotion_ids)="+str(type(all_promotion_ids)))
    print("len(all_promotion_ids)=" + str(len(all_promotion_ids)))
    print("all_promotion_ids=" + str(all_promotion_ids))

    not_in_svd_rec = [x for x in all_promotion_ids if x not in recommended_prod_ids]
    print("len(not_in_svd_rec)="+str(len(not_in_svd_rec)))
    print("not_in_svd_rec=" + str(not_in_svd_rec))
    
    recommended_prod_ids.extend(not_in_svd_rec)

    #Extracting the list of unique promotion names first. This will be an input to be fed in the recommendation function
    # iids = interest_value_for_all_promotions_df['promotion_id'].unique()

    # result_df = pd.DataFrame(iids)
    # result_json = result_df.to_json()
    # print("algo 16. iids="+str(result_json))


    return recommended_prod_ids
