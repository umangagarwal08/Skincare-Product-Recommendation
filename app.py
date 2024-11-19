import streamlit as st # type: ignore
from mainn import recommend_products_with_clustering
from mainn import lst
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


input = st.selectbox("Choose a Product",lst)
get_recommendation=st.button("Guess it")
if get_recommendation:
    if input:
        ft=recommend_products_with_clustering(input)
        st.markdown(ft)
    else:
        st.write("Please choose a product")

