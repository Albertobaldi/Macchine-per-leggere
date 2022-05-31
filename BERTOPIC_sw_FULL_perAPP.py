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
import matplotlib.pyplot as plt
from bertopic import BERTopic

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

st.title("BERTopic – Topic modeling e analisi dei temi su un corpus testuale")

from nltk.corpus import stopwords
stopwords = stopwords.words('italian')

def get_topic_model(lines_pro):
    text = lines_pro
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True)
    topics, probs = topic_model.fit_transform(text)
    freq = topic_model.get_topic_info(); freq.head(5)
    return topics, freq, topic_model
    print(topics)
    print(freq)
            
uploaded_file = st.sidebar.file_uploader("Scegli un file di testo")
st.sidebar.caption('Verifica che il file sia privo di formattazione')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    lines_pro = uploaded_file.read
    get_topic_model(lines_pro, "yes")
