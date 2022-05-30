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
from sklearn.feature_extraction.text import CountVectorizer

def CountVectorizer():
    vectorizer_model = CountVectorizer(stop_words='italian')
    return CountVectorizer()

def topic_model(df):
    text = dataframe
    topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True, vectorizer_model=vectorizer_model)
    topics, _ = topic_model.fit_transform(text)
    return text, topic_model, topics

def get_topics_over_time(text, topics, topic_model):
    topics_over_time = topic_model.topics_over_time(docs=text, 
                                                    topics=topics, 
                                                    global_tuning=True, 
                                                    evolution_tuning=True)
    return topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)

@st.cache(allow_output_mutation=True)
def get_topic_keyword_barcharts(topic_model):
    return topic_model.visualize_barchart(top_n_topics=9, n_words=5, height=800)

uploaded_file = st.sidebar.file_uploader('Carica un file .txt')
st.sidebar.caption('Verifica che il file sia privo di formattazione')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    text = pd.read_table(uploaded_file,header=None)
