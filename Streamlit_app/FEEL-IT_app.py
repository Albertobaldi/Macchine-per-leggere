import streamlit as st
import pandas as pd
import numpy as np
import base64
import warnings
warnings.filterwarnings("ignore")
import re
from tqdm import tqdm
import string
import io
from io import StringIO
import string
from collections import Counter
import matplotlib.pyplot as plt
from feel_it import EmotionClassifier, SentimentClassifier

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
    page_title="FEEL-IT",
    page_icon="🎈",
) 

@st.cache
def get_emotion(file):
  emotion_classifier = EmotionClassifier()
  sent = emotion_classifier.predict(file)
  return sent

@st.cache
def get_sent(file):
  sentiment_classifier = SentimentClassifier()
  em = sentiment_classifier.predict(file)
  return em

uploaded_file = st.sidebar.file_uploader("Scegli un file di testo")
st.sidebar.caption('Verifica che il file sia privo di formattazione')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    filerd = stringio.read()
    listRes = list(filerd.split(" "))
    file = listRes
    if st.button('Processa i dati'):
        get_emotion(file)
        get_sent(file)
        x = [sent]
        keys, counts = np.unique(x, return_counts=True)
        fig1 = plt.bar(keys, counts)
        st.plotly_chart(fig1, use_container_width=True)

        x = [em]
        keys, counts = np.unique(x, return_counts=True)

        fig2 = plt.plot(keys, counts)
        st.plotly_chart(fig2, use_container_width=True)

