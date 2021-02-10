import numpy as np
import pandas as pd

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
    mapping = pd.Series(items.index, index = items)
    index = mapping[item_input]
    
    # get similarity values with other items
    similarity_score = list(enumerate(similarity(data)[index])) # list of index and similarity score
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True) # descending order of scores
    
    # top 10 most similar item names
    similarity_score = similarity_score[1:11] # index 0 is the item itself    
    item_indices = [i[0] for i in similarity_score]
    return items.iloc[item_indices]


def ap_at_k(item):
    item_category = data[['Item Name', 'Category']].drop_duplicates(keep='first').set_index('Item Name')
    to_match = item_category.loc[item][0]
    relevant = []
    for reco in recommend(item):
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