# %%
from bertopic import BERTopic

# %%
from nltk.corpus import stopwords
stopwords = stopwords.words(italian)

#%%

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    string_data = stringio.read()
     st.write(string_data)


# %%
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stopwords)

topic_model = BERTopic(language="multilingual", calculate_probabilities=True, verbose=True, vectorizer_model=vectorizer_model)

# %%
topics, probs = topic_model.fit_transform(uploaded_file)

# %%
freq = topic_model.get_topic_info(); freq.head(5)

# %%
topic_model.get_topic(0)  # Select the most frequent topic

# %%
topic_model.visualize_topics()

# %%
topic_model.visualize_distribution(probs[200], min_probability=0.015)

# %%
topic_model.visualize_hierarchy(top_n_topics=50)

# %%
topic_model.visualize_barchart(top_n_topics=5)

# %%
topic_model.visualize_heatmap(n_clusters=20, width=1000, height=1000)

# %%
topic_model.visualize_term_rank()

# %%
similar_topics, similarity = topic_model.find_topics("amore", top_n=5); similar_topics

# %%
topic_model.get_topic(21)

# %%
topic_model.get_topic(-1)

# %%
topic_model.get_topic(5)

# %%
similar_topics, similarity = topic_model.find_topics("piacere", top_n=5); similar_topics

# %%
topic_model.get_topic(3)

# %%
topic_model.get_topic(6)


