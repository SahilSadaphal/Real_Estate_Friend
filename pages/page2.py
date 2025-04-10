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

st.set_page_config(page_title="Analytics")
st.title("Analytics")
new_df = pd.read_csv("D:\RealEstateProject\data_viz.csv")
with open("D:/RealEstateProject/pages/feature_text.pkl", "rb") as f:
    feature_text = pickle.load(f)


grouped_df = (
    new_df.groupby("sector")[
        ["price", "price_per_sqft", "built_up_area", "latitude", "longitude"]
    ]
    .mean()
    .reset_index()
)
import plotly.express as px

fig = px.scatter_mapbox(
    grouped_df,
    lat="latitude",
    lon="longitude",
    color="price_per_sqft",
    size="built_up_area",
    color_continuous_scale=px.colors.cyclical.IceFire,
    zoom=10,
    mapbox_style="open-street-map",
    hover_name="sector",
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------------------------------------------------#
sector = st.selectbox("Sector", new_df["sector"].sort_values().unique())
wordcloud = pd.read_csv("D:\RealEstateProject\wordclouddf.csv")


def extract_feature(sector):
    main = []
    for item in (
        wordcloud[wordcloud["sector"] == sector]["features"]
        .dropna()
        .apply(ast.literal_eval)
    ):
        main.extend(item)
    return main


main = extract_feature(sector)
feature_text = " ".join(main)

plt.rcParams["font.family"] = "Arial"

wc = WordCloud(
    width=800,
    height=800,
    background_color="white",
    stopwords=set(["s"]),
    min_font_size=10,
).generate(feature_text)
plt.imshow(wc, interpolation="bilinear")
st.pyplot()
