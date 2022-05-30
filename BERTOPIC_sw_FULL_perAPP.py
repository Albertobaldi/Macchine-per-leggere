# %%
from bertopic import BERTopic
import nltk
import streamlit as st
import pandas as pd
nltk.download('stopwords')

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
from nltk.corpus import stopwords
stopwords = stopwords.words("italian")

# %%

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_fwf(uploaded_file)
    st.write(dataframe)


# %%
from sklearn.feature_extraction.text import CountVectorizer

def CountVectorizer():
    vectorizer_model = CountVectorizer(stop_words='italian')

def BERTopic():
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True, vectorizer_model=vectorizer_model)

# %%
def topic_model_transform():
   topics, probs = topic_model.fit_transform(dataframe)

# %%
def topic_model_get_topic_info():
   freq = topic_model.get_topic_info(); freq.head(5)

# %%
def topic_model_get_topic():
   return topic_model.get_topic(0)  # Select the most frequent topic

# %%
def topic_model_visualize_topics():
   return topic_model.visualize_topics()

# %%
def topic_model_visualize_distribution():
   return topic_model.visualize_distribution(probs[200], min_probability=0.015)

# %%
def topic_model_visualize_hierarchy():
   return topic_model.visualize_hierarchy(top_n_topics=50)

# %%
def topic_model_visualize_barchart():
   return topic_model.visualize_barchart(top_n_topics=5)

# %%
def topic_model_visualize_heatmap():
   return topic_model.visualize_heatmap(n_clusters=20, width=1000, height=1000)

# %&
def topic_model_visualize_term_rank():
   return topic_model.visualize_term_rank()
