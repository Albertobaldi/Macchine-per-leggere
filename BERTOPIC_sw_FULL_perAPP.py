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

def get_topic_model():
    text = lines
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True)
    topics, probs = topic_model.fit_transform(lines)
    freq = topic_model.get_topic_info(); freq.head(5)
    return topics, freq, topic_model
    
def topic_model_visualize(topic_model):
    return topic_model.visualize_topics()

def topic_model_distribution(topic_model):
    return topic_model.visualize_distribution(probs[200], min_probability=0.015)

def topic_model_hierarchy(topic_model):
    return topic_model.visualize_hierarchy(top_n_topics=50)

def topic_model_barchart(topic_model):
    return topic_model.visualize_barchart(top_n_topics=5)
            
uploaded_file = st.sidebar.file_uploader("Scegli un file di testo")
st.sidebar.caption('Verifica che il file sia privo di formattazione')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    with open(uploaded_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    

fig1 = topic_model_visualize(get_topic_model)
st.write(fig1)
fig2 = topic_model_distribution(get_topic_model)
st.write(fig2)
fig3 = topic_model_hierarchy(get_topic_model)
st.write(fig3)
fig4 = topic_model_barchart(get_topic_model)
st.write(fig4)
