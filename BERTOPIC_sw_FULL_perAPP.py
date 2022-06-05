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

st.title("BERTopic")
st.subheader("Topic modeling e analisi dei temi su un corpus testuale")

def topic_model(file):
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True)
    return topic_model
            
uploaded_file = st.sidebar.file_uploader("Scegli un file di testo")
st.sidebar.caption('Verifica che il file sia privo di formattazione')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    filerd = stringio.read()
    listRes = list(filerd.split(" "))
    file = listRes

if st.button('Processa i dati'):
    topic_model(file)
    topics, probs = topic_model.fit_transform(file)
    topic_model.get_topic(0)
    topic_model.visualize_topics()
    topic_model.visualize_distribution(probs[200], min_probability=0.015)
    topic_model.visualize_hierarchy(top_n_topics=50)
    topic_model.visualize_barchart(top_n_topics=5) 
