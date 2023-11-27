# da inserire nella pagina 1 streamlit
#import streamlit as st
#from pagina2 import main as pagina2_main
#st.experimental_singleton.add_page("Titolo Pagina 2", pagina2_main)
print('import started')
import streamlit as st
from langdetect import detect
import time
import pandas as pd
import spacy
import gensim
import os
import numpy as np
import hdbscan
import plotly.graph_objs as go
import nltk
nltk.download('omw')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from unidecode import unidecode
import re
#####################################################
print('import ended')
@st.cache_data()
#@st.cache(allow_output_mutation=True, show_spinner=True)
def load_excel(uploaded_file):
    print("Loading Excel file...")  # Debug print
    if uploaded_file is not None:
        print(f"Uploaded file: {uploaded_file.name}")  # Print the file name
        if uploaded_file.name.endswith('.xlsx'):  # Verifica che sia un file Excel
            try:
                # Carica il file Excel con tutti i fogli
                sheets = pd.read_excel(uploaded_file, sheet_name=None)
                print("Excel file loaded successfully.")  # Debug print
                return sheets, ""  # Restituisce i fogli e un messaggio di errore vuoto
            except Exception as e:
                print(f"Error loading Excel file: {e}")  # Print the error
                return None, f"Errore: {e}"  # Restituisce None e il messaggio di errore
        else:
            print("Uploaded file is not an Excel file.")  # Debug print
            return None, "Il file caricato non è un file Excel (.xlsx)."  # Restituisce None e il messaggio di errore
    else:
        print("No file uploaded.")  # Debug print
        return None, "Nessun file caricato."  # Restituisce None e il messaggio di errore

def show_excel_column_headers(sheets, selected_sheet):
    if selected_sheet in sheets:
        return sheets[selected_sheet].columns.tolist()
    return []

def is_file_size_within_limit(uploaded_file, max_size_mb=50):
    if uploaded_file is not None:
        # Convert max size to bytes
        max_size_bytes = max_size_mb * 1024 * 1024
        if uploaded_file.size > max_size_bytes:
            return False
    return True

def select_language():
    valid_languages = ['it', 'en', 'es', 'de', 'fr']
    linguaz = st.selectbox("Scegli la lingua delle keyword:", valid_languages)

    if linguaz == "en":
        stemmer = PorterStemmer()
        sw = stopwords.words('english')
        nlp = spacy.load("en_core_web_lg")
    elif linguaz == "it":
        stemmer = SnowballStemmer("italian")
        sw = stopwords.words('italian')
        nlp = spacy.load("it_core_news_lg")
    elif linguaz == "es":
        stemmer = SnowballStemmer("spanish")
        sw = stopwords.words('spanish')
        nlp = spacy.load("es_core_news_lg")
    elif linguaz == "de":
        stemmer = SnowballStemmer("german")
        sw = stopwords.words('german')
        nlp = spacy.load("de_core_news_lg")
    elif linguaz == "fr":
        stemmer = SnowballStemmer("french")
        sw = stopwords.words('french')
        nlp = spacy.load("fr_core_news_lg")

    return linguaz, stemmer, sw, nlp

#python -m spacy download en_core_web_lg
#python -m spacy download it_core_news_lg

################################################################################################################################
# stampa numero di core della CPU, valore da usare in workers
import multiprocessing
num_cores = multiprocessing.cpu_count()
print(f'Il numero di core CPU disponibili è: {num_cores}')
################################################################################################################################

# Disabilita il badge "Made with Streamlit"
st.set_page_config(layout="wide", page_title="Clusterizer", page_icon=":tada:", initial_sidebar_state="expanded", menu_items={
    'Get Help': 'https://cluster.army',
    'About': "# This is a free SEO tool made by Giovanni Sacheli."
})





print('def ended')


################################################################################################################################

################################################################################################################################

def main():
    print('main started')
    # rimuovi hamburger e firma footer
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden !important;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.title("Clusterizer")
    # Descrizione delle funzionalità del tool
    with st.expander("Click here to expand/hide text", expanded=False):


        st.markdown("""
        ## Key Features
        This tool...
        ## Links
        - [github](https://github.com/evemilano/migtool)
        - [cluster.army](https://cluster.army/)
        - [evemilano.com](https://www.evemilano.com)

        """)

    st.header("Keywords list")
    uploaded_file_2 = st.file_uploader("Upload an Excel file here with your keywords.", type=['xlsx'], key="file_uploader_2")
    if uploaded_file_2 and not is_file_size_within_limit(uploaded_file_2):
        st.error("The file is too large. Please upload a file smaller than 200 MB.")
        uploaded_file_2 = None  # Reset the uploaded file
    elif uploaded_file_2:
        sheets, error_2 = load_excel(uploaded_file_2)
        if sheets:
            selected_sheet_2 = st.selectbox("Select the Excel sheet.", list(sheets.keys()), key="sheet_selector_2")
            column_headers_2 = show_excel_column_headers(sheets, selected_sheet_2)
            selected_column_2 = st.selectbox("Select the columns with Live URLs.", column_headers_2, key="column_selector_2")
            # Chiedi all'utente di inserire le parole chiave da escludere
            exclude_input = st.text_input("Enter keywords to be excluded from the classification, separated by comma:")

            # Ora scegli la lingua e mostra le informazioni
            linguaz, stemmer, sw, nlp = select_language()
            # Usare st.write per mostrare le informazioni
            st.write(f"Language: {linguaz}")
            st.write(f"Stemmer: {stemmer}")
            st.write(f"Stopwords: {sw}")

            # Crea un nuovo DataFrame con solo la colonna selezionata
            selected_sheet_df = sheets[selected_sheet_2]
            df1 = selected_sheet_df[[selected_column_2]]
            # Rinomina la prima colonna di df1 in "Keywords"
            if not df1.empty:
                df1.columns = ["Keywords"]  # Assumendo che vuoi rinominare la prima colonna
            else:
                st.write("Il DataFrame è vuoto.")

            st.write(f'{len(df1)} keywords imported.')
            # pulizia parole chiave

            # Rimuovi le righe con valori nulli o vuoti
            df1 = df1[df1['Keywords'].notnull() & (df1['Keywords'].str.strip() != '')]
            print('righe vuote, Null e NaN rimosse')

            # Togli duplicazioni
            df1.drop_duplicates(subset='Keywords', keep='first', inplace=True)
            print('duplicazioni rimosse')
            
            # Funzione per pulire le parole chiave
            def clean_keyword(keyword):
                # Convertire in minuscolo
                keyword = keyword.lower()
                # Sostituire la punteggiatura con uno spazio
                keyword = re.sub(r'[^\w\s]', ' ', keyword)
                # Rimuovere gli accenti
                keyword = unidecode(keyword)
                return keyword

            # Creare la colonna 'Cleaned'
            df1['Cleaned'] = df1['Keywords'].apply(clean_keyword)








            # Inizializza la variabile di stato
            if 'clustering_completed' not in st.session_state:
                st.session_state['clustering_completed'] = False





            st.table(df1.head(3))
            # bottone avvio!
        
            if st.button("Clusterize!"):
                
                
                
                
                
                # conta caratteri
                df1c=df1.copy()
                # Questa riga sostituisce tutti i valori NaN nella colonna Cleaned con una stringa vuota 
                df1c['Cleaned'] = df1c['Cleaned'].fillna('')
                # Questa riga converte tutti i valori nella colonna Cleaned in stringhe.
                df1c['Cleaned'] = df1c['Cleaned'].apply(str)
                # conteggio
                df1c['Char'] = df1c['Cleaned'].apply(len)

                st.write('colonna conta caratteri creata')
                df2=df1c.copy()
                # assegna alla colonna 'Without stopwords' del dataframe df2 il contenuto della SECONDA (cleaned) colonna del dataframe df, convertendola in stringa
                df2['Without stopwords'] = df2.iloc[:, 0].astype(str)
                # rimuovi le stopwords dalla nuova colonna
                df2['Without stopwords'] = df2['Without stopwords'].apply(
                    lambda x: ' '.join([word for word in x.split() if word.lower() not in sw]))
                st.write('colonna senza stopword creata')
                

                exclude_words = exclude_input.lower().split(",")
                exclude_words = [word.strip() for word in exclude_words]
                
                # Assicurati che df2 sia definito e abbia la colonna 'Without stopwords'
                if 'Without stopwords' in df2.columns:
                    # Crea una copia della colonna "KW without stopwords"
                    df2['Without removed'] = df2['Without stopwords']
                    
                    # Rimuovi le parole chiave escluse dalla nuova colonna
                    df2['Without removed'] = df2['Without removed'].apply(
                        lambda x: ' '.join([word for word in x.split() if word.lower() not in exclude_words]))

                    # Visualizza il DataFrame con la nuova colonna
                    
                    st.write('Column without removed words created.')
                else:
                    st.write("DataFrame df2 does not have the required column 'Without stopwords'")
                
                # stemming
                df3=df2.copy()
                # crea una copia della colonna "KW without stopwords"
                df3['Stemming'] = df3['Without removed']
                # esegui la stemming su ogni parola nella nuova colonna
                df3['Stemming'] = df3['Stemming'].apply(
                    lambda x: ' '.join([stemmer.stem(word) for word in x.split() if word.lower() not in sw]))
                st.write('stemming.')
                
                
                
                #Ordered by frequency
                df4=df3.copy()
                # crea una lista con tutti gli stemmi presenti nella colonna "Long Tail senza stopwords stemmata"
                stem_list = []
                for row in df4['Stemming']:
                    stem_list.extend(row.split())
                # conta le occorrenze di ogni stemma nella lista
                stem_counts = Counter(stem_list)
                # crea una lista di tuple (stemma, frequenza) ordinata in base alla frequenza decrescente
                sorted_stems = sorted(stem_counts.items(), key=lambda x: x[1], reverse=True)
                # crea un dizionario per mappare gli stemmi alle loro posizioni nella lista ordinata
                stem_position_dict = {}
                for i, (stem, _) in enumerate(sorted_stems):
                    stem_position_dict[stem] = i
                # crea la nuova colonna ordinando gli stemmi per frequenza assoluta
                df4['Ordered by frequency'] = df4['Stemming'].apply(
                    lambda x: ' '.join(sorted(x.split(), key=lambda s: stem_position_dict[s])))

                st.write('colonna stemmi ordinati creata')



                #Strongest word stem
                df5=df4.copy()
                # stemma più frequente try ok eng ok it
                # creo una nuova colonna chiamata "Stemma più frequente"
                df5['Strongest word stem'] = df5['Ordered by frequency'].str.split().str[0]

                # Calcola la frequenza di ogni parola in Strongest word stem
                word_freq = df5['Strongest word stem'].value_counts(normalize=True)

                # Calcola il valore percentuale cumulativo di ogni parola in Strongest word stem
                word_cum_pct = word_freq.cumsum()

                # Classifica la forza della parola in Strongest word stem in base al valore percentuale cumulativo
                st.write('classificazione hi med low in corso')
                def classify_word_strength(word):
                    try:
                        if word_cum_pct[word] <= 0.6:  # frequenza alta: "hi"
                            return "hi"
                        elif word_cum_pct[word] <= 0.8:  # frequenza media: "med"
                            return "med"
                        else:  # frequenza bassa: "low"
                            return "low"
                    except KeyError:
                        return "unknown"

                df5['Word Strength'] = df5['Strongest word stem'].apply(classify_word_strength)

                st.write('classificazione hi med low termianta')



                # stemmi
                df6=df5.copy()
                # Crea la nuova colonna "2 Stemmi più frequente" con i due stemmi più frequenti
                df6['Strongest 2 words stems'] = df6['Ordered by frequency'].str.split().str[:2].str.join(' ')
                df6.loc[df6['Strongest 2 words stems'].str.count(' ') < 1, 'Strongest 2 words stems'] = ''
                st.write('classificazione 2x stemmi')
                df7=df6.copy()
                df7['Strongest 3 words stems'] = df7['Ordered by frequency'].str.split().str[:3].str.join(' ')
                df7.loc[df7['Strongest 3 words stems'].str.count(' ') < 2, 'Strongest 3 words stems'] = ''
                st.write('classificazione 3x stemmi')
                df8 = df7.copy()
                df8['Strongest 4 words stems'] = df7['Ordered by frequency'].str.split().str[:4].str.join(' ')
                df8.loc[df8['Strongest 4 words stems'].str.count(' ') < 3, 'Strongest 4 words stems'] = ''
                st.write('classificazione 4x stemmi')
                df8.head(3)
                
                
                
                
                # SPACY
                def process_keywords(df):
                    st.write('Classificazione Spacy per VERBI, NOMI, AGGETTIVI avviata')

                    # Preparazione delle liste per i risultati
                    spacy_cat = []
                    spacy_lemma = []
                    verb_noun_cat = []
                    adj_noun_cat = []
                    vadj_noun_cat = []

                    # Preparazione della barra di progresso
                    progress_bar = st.progress(0)

                    # Analisi di ogni riga del DataFrame con Spacy
                    for index, row in enumerate(df['Keywords']):
                        doc = nlp(row)
                        category = None
                        lemma = None
                        verb_noun = []
                        adj_noun = []
                        vadj_noun = []
                        
                        # Processamento di ogni token nel documento
                        for token in doc:
                            if token.pos_ in ['NOUN']:
                                category = token.text
                                lemma = token.lemma_
                            if token.pos_ in ['VERB', 'NOUN']:
                                verb_noun.append(token.text)
                            if token.pos_ in ['ADJ', 'NOUN']:
                                adj_noun.append(token.text)
                            if token.pos_ in ['ADJ', 'VERB']:
                                vadj_noun.append(token.text)
                        
                        # Gestione dei casi in cui non viene identificata una categoria
                        if category is None:
                            category = 'other'
                        if lemma is None:
                            lemma = 'other'
                        
                        # Aggiunta dei risultati alle liste
                        spacy_cat.append(category)
                        spacy_lemma.append(lemma)
                        verb_noun_cat.append(', '.join(verb_noun))
                        adj_noun_cat.append(', '.join(adj_noun))
                        vadj_noun_cat.append(', '.join(vadj_noun))

                        # Aggiornamento della barra di progresso
                        progress_bar.progress((index + 1) / len(df))

                    # Aggiunta dei risultati al DataFrame
                    df['spacy_cat'] = spacy_cat
                    df['spacy_lemma'] = spacy_lemma
                    df['verb_noun_cat'] = verb_noun_cat
                    df['adj_noun_cat'] = adj_noun_cat
                    df['verb_adj_cat'] = vadj_noun_cat

                    st.write('Classificazione Spacy terminata')
                    return df


                dfspacy = process_keywords(df8)
                st.write(dfspacy.head(5))





                def detect_language(df):
                    st.write('Classificazione LANG iniziata')

                    # Impostazione del batch size
                    batch_size = 1000
                    st.write(f"Inizio della classificazione LANG in blocchi da {batch_size} parole")

                    # Inizializzazione della lista vuota per conservare i risultati
                    langs = []

                    # Calcolo del numero di batch
                    num_batches = len(df) // batch_size + (len(df) % batch_size != 0)

                    # Preparazione della barra di progresso
                    progress_bar = st.progress(0)

                    # Loop attraverso i batch
                    for i in range(num_batches):
                        start_idx = i * batch_size
                        end_idx = (i + 1) * batch_size
                        
                        # Estrai il batch corrente dal DataFrame
                        batch = df['Cleaned'][start_idx:end_idx]
                        
                        # Loop attraverso le parole chiave nel batch
                        for x in batch:
                            if len(x) > 0 and any(c.isalpha() for c in x):
                                try:
                                    lang = detect(x)
                                except:
                                    lang = ''
                            else:
                                lang = ''
                            langs.append(lang)

                        # Aggiornamento della barra di progresso
                        progress_bar.progress((i + 1) / num_batches)

                    # Aggiungi i risultati al DataFrame
                    df['Lang'] = langs

                    st.write('Classificazione LANG completata')
                    return df

                # Utilizzo della funzione nel contesto di Streamlit
                dflang = detect_language(dfspacy)
                st.write(dflang.head(3))
                
                
                # SPACY ELIGIBLE
                def evaluate_similarity(df):
                    st.write('Valutazione della similarità con spaCy iniziata')

                    # Creare un modello di similarità del testo utilizzando tutte le parole chiave
                    keywords_text = ' '.join(df['Keywords'])
                    nlp.max_length = 6000000  # Scegli un numero adeguato al tuo hardware, circa 1 ML di parole
                    keywords_nlp = nlp(keywords_text)

                    # Funzione per valutare l'idoneità di una parola chiave
                    def evaluate_idoneity(keyword):
                        keyword_nlp = nlp(keyword)
                        similarity = keyword_nlp.similarity(keywords_nlp)
                        return 'Eligible' if similarity > 0.2 else 'Not eligible'  # Soglia di similarità, regolabile

                    # Preparazione della barra di progresso
                    progress_bar = st.progress(0)

                    # Applicare la funzione al DataFrame
                    for index, row in enumerate(df['Keywords']):
                        df.at[index, 'NLP Eligible'] = evaluate_idoneity(row)
                        
                        # Aggiornamento della barra di progresso
                        progress_bar.progress((index + 1) / len(df))

                    st.write('Valutazione della similarità completata')
                    return df

                # Utilizzo della funzione nel contesto di Streamlit
                dfelig = evaluate_similarity(dflang)
                st.write("Numero di righe:", len(dfelig))
                st.write(dfelig.head(3))
                
                
                
                # BERT ELIGIBLE
                from sentence_transformers import SentenceTransformer
                from sklearn.metrics.pairwise import cosine_similarity


                def bert_eligible(df):
                    st.write('Caricamento del modello Sentence-BERT')

                    # Caricare un modello Sentence-BERT specifico per l'italiano
                    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

                    st.write('Preparazione dei dati')
                    # Estrai le parole chiave dalla colonna 'Without stopwords' del DataFrame
                    keywords = df['Without stopwords'].tolist()

                    st.write('Generazione degli embeddings')
                    # Generare embeddings per tutte le parole chiave
                    embeddings = model.encode(keywords, convert_to_tensor=True, show_progress_bar=True)

                    st.write('Calcolo della matrice di similarità')
                    # Calcolare la matrice di similarità
                    similarity_matrix = cosine_similarity(embeddings)

                    st.write('Rilevamento delle anomalie')
                    # Identificare le parole chiave anomale
                    mean_similarity = np.mean(similarity_matrix, axis=1)
                    threshold = np.quantile(mean_similarity, 0.1)  # il 10% più basso della similarità media potrebbe essere considerato anomalo
                    df['BERT Eligible'] = ['Eligible' if sim > threshold else 'Not eligible' for sim in mean_similarity]

                    st.write('Valutazione BERT completata')
                    return df

                # Utilizzo della funzione nel contesto di Streamlit
                dfelig = bert_eligible(dfelig)
                st.write("Numero di righe:", len(dfelig))
                st.write(dfelig.head(3))




                # dbscan cluster


                from gensim.models import Word2Vec
                from sklearn.preprocessing import StandardScaler

                # Assumi che dfelig sia il DataFrame che desideri processare

                def hdbscan_cluster(df):
                    st.write('Preparazione dei dati per il modello Word2Vec')

                    # Preparazione dei dati per l'addestramento del modello Word2Vec
                    df['Split_Words'] = df['Cleaned'].apply(lambda x: x.split())

                    # Addestramento del modello Word2Vec
                    st.write('Addestramento del modello Word2Vec')
                    model = Word2Vec(sentences=df['Split_Words'], vector_size=100, window=5, min_count=1, workers=4)
                    model.train(df['Split_Words'], total_examples=len(df['Split_Words']), epochs=10)

                    # Trasformazione delle parole chiave in vettori
                    st.write('Trasformazione delle parole chiave in vettori')
                    def phrase_to_vec(phrase):
                        words = phrase.split()
                        vectors = [model.wv[word] for word in words if word in model.wv.index_to_key]
                        if vectors:
                            return np.mean(vectors, axis=0)
                        else:
                            return np.nan

                    df['Vector'] = df['Cleaned'].apply(phrase_to_vec)

                    # Rimozione delle righe dove la frase non può essere convertita in vettore
                    df.dropna(subset=['Vector'], inplace=True)

                    # Trasformazione dei vettori in un array NumPy per HDBSCAN
                    st.write('Preparazione dei dati per HDBSCAN')
                    vectors = np.stack(df['Vector'].to_numpy())

                    # Standardizzazione delle caratteristiche
                    scaler = StandardScaler()
                    vectors = scaler.fit_transform(vectors)

                    # Applicazione del clustering HDBSCAN
                    st.write('Applicazione del clustering HDBSCAN')
                    clustering = hdbscan.HDBSCAN(min_samples=2).fit(vectors)

                    # Aggiunta delle etichette al DataFrame
                    df['Word2Vec_HDBSCAN Cluster'] = clustering.labels_

                    st.write('Clustering completato')
                    return df

                # Utilizzo della funzione nel contesto di Streamlit
                dfdbscan = hdbscan_cluster(dfelig)
                st.write("Numero di righe:", len(dfdbscan))
                st.write(dfdbscan.head(3))








                # BERT HDBSCAN TORCH
                from transformers import AutoTokenizer, AutoModel
                import torch

                dfbert = dfdbscan.copy()

                # Caricare il modello BERT pre-addestrato per l'italiano
                model_name = "dbmdz/bert-base-italian-xxl-cased"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)

                # Preparazione dei dati per BERT
                st.write("Preparazione dei dati per BERT")

                # Tokenizzazione e creazione degli embeddings
                keywords = dfbert['Without stopwords'].tolist()
                tokens = tokenizer(keywords, padding=True, truncation=True, return_tensors="pt")
                input_ids = tokens['input_ids']
                attention_mask = tokens['attention_mask']

                # Definire la dimensione del batch
                batch_size = 1000  # Aggiusta questo numero in base alla capacità della tua CPU
                embeddings = []

                progress_bar = st.progress(0)
                total_size = input_ids.size(0)
                for i in range(0, total_size, batch_size):
                #for i in range(0, input_ids.size(0), batch_size):
                    batch_input_ids = input_ids[i:i+batch_size]
                    batch_attention_mask = attention_mask[i:i+batch_size]

                    with torch.no_grad():
                        batch_outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                        batch_embeddings = batch_outputs.last_hidden_state.mean(dim=1)
                        embeddings.append(batch_embeddings)

                    # Aggiorna la barra di progresso
                    #progress_bar.progress(min((i+batch_size) / input_ids.size(0), 1))
                    progress = (i+batch_size) / total_size
                    progress_bar.progress(min(progress, 1.0))
                
                progress_bar.progress(1.0)
  
                # Concatenare gli embeddings per ottenere un unico tensore
                embeddings = torch.cat(embeddings, dim=0)

                # Applicare HDBSCAN per il clustering
                st.write('Applicazione del clustering HDBSCAN')
                clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True, core_dist_n_jobs=-1)
                dfbert['Bert_Hdbscan Cluster'] = clusterer.fit_predict(embeddings.numpy())

                # Mostrare il risultato del clustering
                st.write("Clustering completato")
                st.write(dfbert.head(3))




                # assegnazione frasi ai numedi dei cluster BERT



                def assign_phrases_to_clusters(df, embeddings):
                    st.write("Inizio assegnazione delle frasi ai numeri dei cluster")

                    # Funzione per trovare la frase più centrale in un cluster
                    def find_central_phrase(phrases, cluster_embeddings):
                        similarity_matrix = cosine_similarity(cluster_embeddings)
                        centrality_scores = similarity_matrix.sum(axis=1)
                        return phrases[np.argmax(centrality_scores)]

                    cluster_names = {}

                    progress_bar = st.progress(0)
                    total_clusters = df['Bert_Hdbscan Cluster'].nunique()
                    processed_clusters = 0

                    for cluster_id in df['Bert_Hdbscan Cluster'].unique():
                        cluster_data = df[df['Bert_Hdbscan Cluster'] == cluster_id]
                        phrases = cluster_data['Without stopwords'].tolist()
                        cluster_embeddings = embeddings[cluster_data.index]

                        central_phrase = find_central_phrase(phrases, cluster_embeddings)

                        while central_phrase in cluster_names.values():
                            phrases.remove(central_phrase)
                            if not phrases:
                                break
                            cluster_embeddings = embeddings[cluster_data[cluster_data['Without stopwords'].isin(phrases)].index]
                            central_phrase = find_central_phrase(phrases, cluster_embeddings)

                        cluster_names[cluster_id] = central_phrase

                        processed_clusters += 1
                        progress_bar.progress(processed_clusters / total_clusters)

                    df['Cluster Name'] = df['Bert_Hdbscan Cluster'].map(cluster_names)

                    return df

                # Utilizza la funzione nel contesto di Streamlit
                dfbert = assign_phrases_to_clusters(dfbert, embeddings)
                st.write("Assegnazione completata")
                st.dataframe(dfbert.head(3))
                
                
         
         


                



                st.session_state['clustering_completed'] = True
                # Verifica se il clustering è stato completato
                if st.session_state['clustering_completed']:
                    # Qui puoi inserire il codice per il bottone di download
                    # Esempio:
                    st.download_button(label="Scarica il csv", data=dfbert.to_csv().encode('utf-8'), file_name='clustered_data.csv', mime='text/csv')
         
                import io
                from io import BytesIO






                # Prepara un file Excel in memoria
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    dfbert.to_excel(writer, sheet_name='Sheet1')
                    # Nota: non è necessario chiamare writer.save()

                # Riporta il puntatore all'inizio del file
                output.seek(0)

                # Crea il bottone di download
                st.download_button(
                    label="Download Excel file",
                    data=output,
                    file_name="dfbert.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

         
         
         


                         
        else:
            st.write(error_2)






    
if __name__ == "__main__":
    main()