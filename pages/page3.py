import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast

st.set_option("deprecation.showPyplotGlobalUse", False)

st.set_page_config(page_title="Recommendtation")
st.title("Recommendation Module")
with open("D:\RealEstateProject\my_dataframe_norm.pkl", "rb") as f:
    df = pickle.load(f)
with open("D:\RealEstateProject\cosine_sim1", "rb") as f:
    cosine_sim1 = pickle.load(f)
with open("D:\RealEstateProject\cosine_sim2", "rb") as f:
    cosine_sim2 = pickle.load(f)
with open("D:\RealEstateProject\cosine_sim3", "rb") as f:
    cosine_sim3 = pickle.load(f)


property_selected = st.selectbox("Property", df.index.to_list())
if st.button("Recommend"):

    def recommend_properties_with_scores(property_name, top_n=247):
        cosine_sim_matrix = 30 * cosine_sim1 + 20 * cosine_sim2 + 8 * cosine_sim3
        # cosine_sim_matrix = cosine_sim3

        # Get the similarity scores for the property using its name as the index
        sim_scores = list(enumerate(cosine_sim_matrix[df.index.get_loc(property_name)]))

        # Sort properties based on the similarity scores
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices and scores of the top_n most similar properties
        top_indices = [i[0] for i in sorted_scores[1 : top_n + 1]]
        top_scores = [i[1] for i in sorted_scores[1 : top_n + 1]]

        # Retrieve the names of the top properties using the indices
        top_properties = df.index[top_indices].tolist()

        # Create a dataframe with the results
        recommendations_df = pd.DataFrame(
            {"PropertyName": top_properties, "SimilarityScore": top_scores}
        )

        return recommendations_df

    # Test the recommender function using a property name
    st.dataframe(recommend_properties_with_scores(property_selected))
