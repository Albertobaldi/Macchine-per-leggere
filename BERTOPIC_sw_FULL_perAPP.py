# %%
from bertopic import BERTopic
import nltk
import streamlit as st
import pandas as pd
nltk.download('stopwords')

# %%
from nltk.corpus import stopwords
stopwords = stopwords.words('italian')

# %%
st.set_page_config(
    page_title="BERTopic",
    page_icon="🎈",
)

# %%

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
    
# %%
uploaded_file = st.sidebar.file_uploader('Carica un file .txt')
st.sidebar.caption('Verifica che il file sia privo di formattazione')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    text = pd.read_table(uploaded_file,header=None)
