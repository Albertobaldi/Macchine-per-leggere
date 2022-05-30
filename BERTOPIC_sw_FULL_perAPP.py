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
    dataframe = pd.read_table(uploaded_file,header=None)
    
# %%
from sklearn.feature_extraction.text import CountVectorizer

def CountVectorizer():
    vectorizer_model = CountVectorizer(stop_words='italian')
    return CountVectorizer()

def get_topic_model(df):
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

df = None
uploaded_file = st.sidebar.file_uploader('Choose a CSV file')
st.sidebar.caption('Make sure the csv contains a column titled "date" and a column titled "text"')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    print(df.head().to_markdown())
    # st.write(df)
elif st.sidebar.button('Load demo data'):
    data_load_state = st.text('Loading data...')
    df = pd.read_csv('./cleaned_data/medium-suggested-cleaned.csv')
    data_load_state.text('Loading data... done!')
    if st.checkbox('Preview the data'):
        st.subheader('5 rows of raw data')
        st.write(data[:5])

    # st.write(df.head())

if df is not None:
    # concatenate title and subtitle columns
    data_clean_state = st.text('Cleaning data...')
    df['text'] = preprocess(df['text'].astype(str))
    cleaned_df = df[['date', 'text']]
    cleaned_df = cleaned_df.dropna(subset=['text'])
    st.write(len(cleaned_df), "total documents")
    data_clean_state.text('Cleaning data... done!')

    tm_state = st.text('Modeling topics...')
    text, topic_model, topics = get_topic_model(cleaned_df)
    tm_state.text('Modeling topics... done!')

    freq = topic_model.get_topic_info(); 
    st.write(freq.head(10))

    fig1 = get_intertopic_dist_map(topic_model)
    st.write(fig1)

    fig2 = get_topics_over_time(text, topics, topic_model)
    st.write(fig2)

    fig3 = get_topic_keyword_barcharts(topic_model)
    st.write(fig3)
