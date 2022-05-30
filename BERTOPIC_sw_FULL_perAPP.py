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

def preprocess(text_col):
    """This function will apply NLP preprocessing lambda functions over a pandas series such as df['text'].
       These functions include converting text to lowercase, removing emojis, expanding contractions, removing punctuation,
       removing numbers, removing stopwords, lemmatization, etc."""
    
    # convert to lowercase
    text_col = text_col.apply(lambda x: ' '.join([w.lower() for w in x.split()]))
    
    # remove emojis
    text_col = text_col.apply(lambda x: demoji.replace(x, ""))
    
    # expand contractions  
    text_col = text_col.apply(lambda x: ' '.join([contractions.fix(word) for word in x.split()]))

    # remove punctuation
    text_col = text_col.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
    
    # remove numbers
    text_col = text_col.apply(lambda x: ' '.join(re.sub("[^a-zA-Z]+", " ", x).split()))

    # remove stopwords
    stopwords = [sw for sw in list(nltk.corpus.stopwords.words('english')) + ['thing'] if sw not in ['not']]
    text_col = text_col.apply(lambda x: ' '.join([w for w in x.split() if w not in stopwords]))

    # lemmatization
    text_col = text_col.apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(w) for w in x.split()]))

    # remove short words
    text_col = text_col.apply(lambda x: ' '.join([w.strip() for w in x.split() if len(w.strip()) >= 3]))

    return text_col

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
uploaded_file = st.sidebar.file_uploader('Carica un file .txt')
st.sidebar.caption('Verifica che il file sia privo di formattazione')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    text = pd.read_table(uploaded_file,header=None)
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
