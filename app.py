"""
Recommender System using SVD
"""

import numpy as np
import pandas as pd
import streamlit as st
from svd import convert_rating, svd_recommendation, recommend

df = pd.read_csv('NewProfile.csv')

def welcome():
    return "Welcome All"

def predict_reco(data, user_id, new_min, new_max):
    
    """predict product recommendations
    ---
    parameters follow the svd recommend function:
      - data (dataframe)
      - user_id (string)
      - new_min (integer)
      - new_max (integer) 
    output:
        json object of products recommended and respective predicted ratings       
    """
   
    reco = recommend(data, user_id, new_min, new_max)
    return reco

def main():
    st.title("Product Recommendations")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Product Recommender App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    user_id = st.text_input("User Id","Type Here")
    new_min = st.text_input("Minimum Rating","Type Here")
    new_max = st.text_input("Maximum Rating","Type Here")
    result=""
    if st.button("Recommend"):
        result=predict_reco(df, str(user_id), int(new_min), int(new_max))
    st.success('The recommendations are: {}'.format(result))

if __name__=='__main__':
    main()
    #user_id = 'A3SGXH7AUHU8GW'
    #print(predict_reco(df, user_id, 1, 5))