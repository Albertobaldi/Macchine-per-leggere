# %%
from bertopic import BERTopic
import nltk
import streamlit as st
import pandas as pd
from io import StringIO
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

def get_topic_model(data):
    text = data
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True)
    topics, probs = topic_model.fit_transform(file)
    freq = topic_model.get_topic_info(); freq.head(5)
    return text, topics, freq

# %%

def topic_model_visualize(topic_model):
    return topic_model.visualize_topics()

# %%

def topic_model_distribution(topic_model):
    return topic_model.visualize_distribution(probs[200], min_probability=0.015)

# %%

def topic_model_hierarchy(topic_model):
    return topic_model.visualize_hierarchy(top_n_topics=50)
    
# %%

def topic_model_barchart(topic_model):
    return topic_model.visualize_barchart(top_n_topics=5)

# %%

file = None
uploaded_file = st.sidebar.file_uploader('Carica un file .txt')
st.sidebar.caption('Verifica che il file sia privo di formattazione')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    file = stringio.read()
    st.write(file)
    text, topic_model, topics = get_topic_model(data)
    
    fig1 = topic_model_visualize(topic_model)
    st.write(fig1)
    
    fig2 = topic_model_distribution(topic_model)
    st.write(fig2)
    
    fig3 = topic_model_hierarchy(topic_model)
    st.write(fig3)
    
    fig4 = topic_model_barchart(topic_model)
    st.write(fig4)
