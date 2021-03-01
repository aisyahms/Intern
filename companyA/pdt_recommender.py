import numpy as np
import pandas as pd
import math

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_excel('edit-CompanyA-RetailCorpus.xlsx')
items = pd.Series(data['Item Name'].unique())

def similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    item_matrix = tfidf.fit_transform(items)
    similarity_matrix = linear_kernel(item_matrix,item_matrix)
    return similarity_matrix


def recommend(item_input):
    reco = {}
    mapping = pd.Series(items.index, index = items)
    index = mapping[item_input]
    
    # get similarity values with other items
    similarity_score = list(enumerate(similarity(data)[index])) # list of index and similarity score
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True) # descending order of scores
    
    # top 10 most similar item names
    similarity_score = similarity_score[1:11] # index 0 is the item itself    
    item_indices = [i[0] for i in similarity_score]
    for idx in range(len(item_indices)):
        reco[items.iloc[item_indices[idx]]] = similarity_score[idx][1] # assign the similarity score
    return reco


def ap_at_k(item):
    """Relevance: recommended item is within the same category as item"""
    item_category = data[['Item Name', 'Category']].drop_duplicates(keep='first').set_index('Item Name')
    to_match = item_category.loc[item][0]
    relevant = []
    for reco in recommend(item).keys():
        reco_cat = item_category.loc[reco][0]
        if to_match == reco_cat:
            relevant.append(1)
        else:
            relevant.append(0)
#     print(relevant) 
    
    precision_sum = 0
    relevant_count = 0
    for i in range(len(relevant)):
        if relevant[i] == 1:
            relevant_count += 1
            precision_sum += relevant_count/(i+1)
    if relevant_count == 0:
        score = 0
    else:
        score = precision_sum/relevant_count
    return score


data['Label'] = data.groupby(['Outlet','Order Date']).grouper.group_info[0]
data.drop_duplicates(subset=['Label','Item Name'], keep='first', inplace=True)
label_dict = {}
for i in range(len(data)):
    key = data.iloc[i]['Label']
    if key in label_dict:
        label_dict[key].append(data.iloc[i]['Item Name'])
    else:
        label_dict[key] = [data.iloc[i]['Item Name']]
        
def ap_at_k2(item):
    """Relevance: recommended item was indeed bought under one of the same (Outlet, Order Date) labels as item"""
    labels = [x for x in data[data['Item Name'] == item]['Label'].unique()] # labels with the item purchased
    bought_tgt = [] # items in the same label/ transaction with item
    for l in labels:
        bought_tgt += label_dict[l]
    bought_tgt = set(bought_tgt)
    items_reco = list(recommend(item).keys())
#     print(items_reco)
    
    precision_sum = 0
    relevant_count = 0
    for i in range(len(items_reco)):
        if items_reco[i] in bought_tgt:
            relevant_count += 1
            precision_sum += relevant_count/(i+1) 
    if relevant_count == 0:
        score = 0
    else:
        score = precision_sum/relevant_count
    return score


def mean_pmi(item):
    """i: item
    j: item being paired with i"""
    labels = [x for x in data[data['Item Name'] == item]['Label'].unique()] # labels with the input item purchased
    bought_tgt = [] # items in the same label/ transaction with item
    for l in labels:
        bought_tgt += label_dict[l]
     
    num_labels = data['Label'].nunique()
    prob_i = data[data['Item Name'] == item]['Label'].nunique()/num_labels
    total_ij = 0
    
    items_reco = list(recommend(item).keys())
    for i in range(len(items_reco)):
        if items_reco[i] in bought_tgt:
            prob_j = data[data['Item Name'] == items_reco[i]]['Label'].nunique()/num_labels
            num_ij = bought_tgt.count(items_reco[i])
            prob_ij = num_ij/num_labels
            total_ij += (-math.log((prob_ij/(prob_i*prob_j)),2))/math.log(prob_ij,2)   
        # else add 0
    
    return total_ij/10