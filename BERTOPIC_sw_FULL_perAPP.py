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
            
uploaded_file = st.sidebar.file_uploader("Scegli un file di testo")
st.sidebar.caption('Verifica che il file sia privo di formattazione')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    filerd = stringio.read()
    listRes = list(filerd.split(" "))
    file = listRes

if st.button('Processa i dati'):
    st.write("Il vostro file è in elaborazione. Il tempo impiegato nell’analisi dei topic può variare a seconda delle dimensioni del file di testo.")
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True, vectorizer_model=vectorizer_model)
    topics, probs = topic_model.fit_transform(file)
    freq = topic_model.get_topic_info(); freq.head(5)
    get = topic_model.get_topics()
    fig = topic_model.visualize_topics()
    info = topic_model.get_topic_info()
    st.table(get)
    st.table(info)
    st.plotly_chart(fig, use_container_width=False)
