import streamlit as st
import pandas as pd
import numpy as np
import base64
import warnings
warnings.filterwarnings("ignore")
import re
from tqdm import tqdm
import string
import nltk
nltk.download('stopwords')
import io
from io import StringIO
import string
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from bertopic import BERTopic
import itertools
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

st.set_page_config(
    page_title="BERTopic",
    page_icon="🎈",
) 

st.title("BERTopic")
st.subheader("Topic modeling e analisi dei temi su un corpus testuale")

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

final_stopwords_list = stopwords.words('italian')
tfidf_vectorizer = TfidfVectorizer(max_df=0.5,
  max_features=200000,
  stop_words=final_stopwords_list,
  use_idf=True)

vectorizer_model = tfidf_vectorizer

def get_topic_model(file):
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True, vectorizer_model=vectorizer_model)
    topics, probs = topic_model.fit_transform(file)
    freq = topic_model.get_topic_info(); freq.head(5)
    return topics, freq, topic_model

def visualize_distribution(topic_model,
                           probabilities: np.ndarray,
                           min_probability: float = 0.015,
                           width: int = 800,
                           height: int = 600) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        marker=dict(
            color='#C8D2D7',
            line=dict(
                color='#6E8484',
                width=1),
        ),
        orientation='h')
    )

    fig.update_layout(
        xaxis_title="Probability",
        title={
            'text': "<b>Topic Probability Distribution",
            'y': .95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )
    return fig
            
uploaded_file = st.sidebar.file_uploader("Scegli un file di testo")
st.sidebar.caption('Verifica che il file sia privo di formattazione')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    filerd = stringio.read()
    listRes = list(filerd.split(" "))
    file = listRes

if st.button('Processa i dati'):
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True, vectorizer_model=vectorizer_model)
    topics, probs = topic_model.fit_transform(file)
    freq = topic_model.get_topic_info(); freq.head(5)
    get = topic_model.get_topic(0)
    topic_model.visualize_distribution(probs[200], min_probability=0.015)
    st.plotly_chart(fig, use_container_width=True)
