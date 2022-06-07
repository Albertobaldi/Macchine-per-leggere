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

nltk.download('stopwords')

sw = st.sidebar.text_input("Inserisci una lista di stopwords, tra apici doppi e separate da una virgola", "", placeholder="\"parola1\", \"parola2\", \"parola3\"")
final_stopwords_list = sw
vectorizer_model = CountVectorizer(stop_words = sw)

@st.cache
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
    file = stringio.read().split('\n')

if st.button('Processa i dati'):
    st.write("Il vostro file è in elaborazione. Il tempo impiegato nell’analisi dei topic può variare a seconda delle dimensioni del file di testo.")
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True, vectorizer_model=vectorizer_model)
    topics, probs = topic_model.fit_transform(file)
    freq = topic_model.get_topic_info(); freq.head(10)
    info = topic_model.get_topic_info()
    top = topic_model.visualize_barchart(top_n_topics=10)
    distribution = topic_model.visualize_distribution(probs[100], min_probability=0.015)
    st.write(info)
    st.plotly_chart(top, use_container_width=True)
    st.plotly_chart(distribution, use_container_width=True)
