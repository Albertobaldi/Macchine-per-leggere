import streamlit as st
import warnings
warnings.filterwarnings("ignore")
from io import StringIO
from sentence_transformers import SentenceTransformer, util
import torch
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
st.title("Ricerca semantica")
st.subheader("Interroga un romanzo ricercando per frasi o sintagmi simili")
            
uploaded_file = st.sidebar.file_uploader("Scegli un file di testo")
st.sidebar.caption('Verifica che il file sia privo di formattazione. Si raccomanda di convertire ogni fine di paragrafo in interruzione di linea (\\n): così facendo, l’algoritmo potrà suddividere il testo in paragrafi')
st.sidebar.markdown("""---""")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    file = stringio.read().split('\n')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = embedder.encode(file, convert_to_tensor=True)
if st.text_input('Inserisci una frase'):
    queries = st.text_input
    top_k = min(5, len(file))
    query_embedding = embedder.encode(queries, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores 
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    st.write("\n\n======================\n\n")
    st.write("Query:", query)
    st.write("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        st.write(corpus[idx], "(Score: {:.4f})".format(score))
