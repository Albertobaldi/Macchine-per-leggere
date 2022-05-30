from bertopic import BERTopic
import nltk
import streamlit as st
import pandas as pd
import io
from io import StringIO
import string
from collections import Counter

from nltk.corpus import stopwords
stopwords = stopwords.words('italian')

st.set_page_config(
    page_title="BERTopic",
    page_icon="🎈",
)

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
    
def get_topic_model(file):
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True)
    topics, probs = topic_model.fit_transform(file)
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
    
filename = None
uploaded_file = st.sidebar.file_uploader('Carica un file .txt')
st.sidebar.caption('Verifica che il file sia privo di formattazione')
st.sidebar.markdown("""---""")
if filename is not None:
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()    

    text, dates, topic_model, topics = get_topic_model(lines)
    
with st.container():
    st.write("This is inside the container")

    fig1 = topic_model_visualize(topic_model)
    st.write(fig1)
    
    fig2 = topic_model_distribution(topic_model)
    st.write(fig2)
    
    fig3 = topic_model_hierarchy(topic_model)
    st.write(fig3)
    
    fig4 = topic_model_barchart(topic_model)
    st.write(fig4)
