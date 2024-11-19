import streamlit as st # type: ignore
from mainn import recommend_products_with_clustering
from mainn import lst



input = st.selectbox("Choose a Product",options=lst,index=None)
get_recommendation=st.button("Guess it")
if get_recommendation:
    if input:
        ft=recommend_products_with_clustering(input)
        st.markdown(ft)
    else:
        st.write("Please choose a product")

