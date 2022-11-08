import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from io import StringIO
from bertopic import BERTopic
from bertopic.backend import languages
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
from sklearn.feature_extraction.text import CountVectorizer
if 'final_stopwords' not in st.session_state:
	st.session_state.final_stopwords = False
final_stopwords_list = st.sidebar.text_input("Inserisci una lista di stopwords, senza lettere maiuscole (neppure nei nomi propri) e separate da una virgola (es. \"parola1, parola2, parola3\")", "").split(', ')
        
vectorizer_model = CountVectorizer(stop_words=final_stopwords_list)
            
uploaded_file = st.sidebar.file_uploader("Scegli un file di testo")
st.sidebar.caption('Verifica che il file sia privo di formattazione. Si raccomanda di convertire ogni fine di paragrafo in interruzione di linea (\\n): così facendo, l’algoritmo potrà suddividere il testo in paragrafi')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    file = stringio.read().split('\n')
if st.button('Processa i dati per visualizzare la distribuzione dei topic nel corpus'):
    st.write("Il vostro file è in elaborazione. Il tempo impiegato nell’analisi dei topic può variare a seconda delle dimensioni del file di testo.")
    topic_model = BERTopic(language="italian", top_n_words=10, calculate_probabilities=True, verbose=True, vectorizer_model=vectorizer_model, min_topic_size=3, nr_topics=10)
    topics, probs = topic_model.fit_transform(file)
    freq = topic_model.get_topic_info(); freq.head(10)
    info = topic_model.get_topic_info()
    fig1 = topic_model.visualize_topics()
    fig2 = topic_model.visualize_distribution(probs[0], min_probability=0)
    fig3 = topic_model.visualize_hierarchy(top_n_topics=10)
    fig4 = topic_model.visualize_barchart(top_n_topics=10)
    fig5 = topic_model.visualize_heatmap(n_clusters=9, width=1000, height=1000)
    fig6 = topic_model.visualize_term_rank()
    st.write(info)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)
    st.plotly_chart(fig5, use_container_width=True)
    st.plotly_chart(fig6, use_container_width=True)
