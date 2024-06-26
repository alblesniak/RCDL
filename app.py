import streamlit as st
from hydralit import HydraApp
from keywords import load_keywords_data
import logging
import time
import uuid
from itertools import groupby
from keywords import load_keywords_data
from collocations import (compute_log_likelihood_scores_batch_partitioned,
                          filter_tokens_by_doc_ids, find_collocations,
                          find_relevant_doc_ids, find_detailed_collocation_occurrences, get_context_for_collocations, generate_markdown_report)
from user_input_functions import get_user_input_collocations, get_user_input_concordances
from annotated_text import annotated_text

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

st.set_page_config(layout="wide", page_title='Religious Cultures DataLab')

app = HydraApp(title='Religious Cultures DataLab', use_loader=False, hide_streamlit_markers=True, sidebar_state='expanded')

# Funkcja resetująca stan sesji
def reset_session_state():
    st.session_state.pop('selected_journals', None)
    st.session_state.pop('year_range', None)
    st.session_state.pop('query_lemma', None)
    st.session_state.pop('search_type', None)
    st.session_state.pop('left_context_length', None)
    st.session_state.pop('right_context_length', None)
    st.session_state.pop('n', None)
    st.session_state.pop('is_stop', None)
    st.session_state.pop('is_punct', None)
    st.session_state.pop('selected_pos', None)
    st.session_state.pop('sort_by', None)
    st.session_state.pop('sample_percentage', None)

# Strona tytułowa
@app.addapp(title='', is_home=True, icon="fas fa-home")
def home_page():
    st.markdown("### Strona Tytułowa")

@app.addapp(title='Słowa kluczowe')
def keyword_page():
    reset_session_state()  # Resetuj stan sesji
    options = ['Katolickie', 'Pentekostalne']
    selected_corpus = st.sidebar.selectbox('Wybierz korpus:', options, key='selected_corpus')

    merged_df = load_keywords_data(selected_corpus)
    merged_df = merged_df.sort_values(by='log_likelihood', ascending=False)

    st.dataframe(
        merged_df,
        column_config={
            "keyword": st.column_config.TextColumn("Słowo kluczowe", width="medium"),
            "log_likelihood": st.column_config.NumberColumn("Log Likelihood"),
            "occurrences_K": st.column_config.NumberColumn("Liczba wystąpień (K)"),
            "occurrences_per_1000_K": st.column_config.NumberColumn("Wystąpienia na 1000 słów (K)"),
            "occurrences_P": st.column_config.NumberColumn("Liczba wystąpień "),
            "occurrences_per_1000_P": st.column_config.NumberColumn("Wystąpienia na 1000 słów (P)"),
            "occurrences_over_time": st.column_config.LineChartColumn("Wystąpienia w czasie"),
        },
        hide_index=True,
        width=1500,
        height=600
    )

@app.addapp(title='Kolokacje')
def collocation_page():
    reset_session_state()  # Resetuj stan sesji
    if 'context_occurrences' not in st.session_state:
        st.session_state.context_occurrences = []

    user_inputs = get_user_input_collocations()

    if st.sidebar.button('Wyszukaj kolokacje'):
        (selected_journals, year_range, query_lemma, search_type,
         left_context_length, right_context_length, n,
         is_stop, is_punct, selected_pos) = user_inputs
        if query_lemma:
            with st.spinner("Wyszukiwanie danych tekstowych dla wybranych kryteriów"):
                start_time = time.time()
                relevant_doc_ids = find_relevant_doc_ids(selected_journals, year_range)
                filtered_df = filter_tokens_by_doc_ids(relevant_doc_ids)
                logger.info(f"Wyszukiwanie kolokacji dla segmentu '{query_lemma}' w podanym zakresie")
                start_time_collocations = time.time()
                collocations_result = find_collocations(filtered_df, query_lemma, search_type,
                                                        left_context_length, right_context_length,
                                                        is_stop, is_punct, selected_pos)
                time_for_collocations = time.time() - start_time_collocations
                logger.info(f"Znaleziono kolokacji: {len(collocations_result)}, czas wykonania: {time_for_collocations:.2f}s.")

                total_tokens = len(filtered_df)
                log_likelihood_scores = compute_log_likelihood_scores_batch_partitioned(collocations_result, total_tokens, search_type)

                top_collocations_with_details = []
                for collocation, (count, doc_ids) in collocations_result.items():
                    score = log_likelihood_scores.get(collocation, 0)
                    top_collocations_with_details.append((collocation, score, doc_ids))
                top_collocations_with_details.sort(key=lambda item: item[1], reverse=True)
                top_collocations_with_details = top_collocations_with_details[:n]

                detailed_occurrences = find_detailed_collocation_occurrences(
                    top_collocations_with_details=top_collocations_with_details,
                    tokens_df=filtered_df,
                    query_lemma=query_lemma,
                    search_type=search_type,
                    left_context_length=left_context_length,
                    right_context_length=right_context_length,
                    is_stop=is_stop,
                    is_punct=is_punct,
                    selected_pos=selected_pos
                )

                st.session_state.context_occurrences = get_context_for_collocations(detailed_occurrences, filtered_df)
                elapsed_time = time.time() - start_time

                logger.info(f"Całkowity czas wykonania: {elapsed_time}")

    if st.session_state.context_occurrences:
        st.write("Najsilniejsze kolokacje z kontekstem:")
        st.session_state.context_occurrences.sort(key=lambda x: x['score'], reverse=True)

        for key, group in groupby(st.session_state.context_occurrences, key=lambda x: x['collocation']):
            group_list = list(group)
            score = group_list[0]['score']
            expander_title = f"{key} - {score}"
            with st.expander(expander_title, expanded=False):
                for context_data in group_list:
                    annotated_content = []
                    for idx, item in enumerate(context_data['context']):
                        if idx > 0:
                            prev_item = context_data['context'][idx - 1]
                            if isinstance(prev_item, str) and not isinstance(item, tuple):
                                annotated_content.append(' ')
                        annotated_content.append(item)
                    annotated_text(*annotated_content)
                    st.write("")  # Optional spacing for better readability

    if st.session_state.context_occurrences:
        markdown_content = generate_markdown_report(user_inputs, st.session_state.context_occurrences)
        unique_key = str(uuid.uuid4())
        st.download_button(label="Download Report as Markdown",
                           data=markdown_content,
                           file_name="collocation_report.md",
                           mime="text/markdown",
                           key=unique_key)

@app.addapp(title='Konkordancje')
def concordance_page():
    reset_session_state()  # Resetuj stan sesji
    user_inputs = get_user_input_concordances()

    if st.sidebar.button('Wyszukaj konkordancje'):
        selected_journals, year_range, query_lemma, search_type, left_context_length, right_context_length, sort_by, sample_percentage = user_inputs
        st.write(user_inputs)

# Moduł Wektory Słów
@app.addapp(title='Wektory słów')
def word_vectors_page():
    st.sidebar.markdown("## Osadzenia słów dla polskiej prasy wyznaniowej oraz tygodników opinii")
    st.markdown("### Hello Word Vectors")

@app.addapp(title='Modele tematyczne')
def topic_models_page():
    st.sidebar.markdown("## Wizualizacje tematów")
    st.markdown("### Hello Topic Models")

# Moduł Wyszukiwanie Semantyczne
@app.addapp(title='Wyszukiwanie semantyczne')
def semantic_search_page():
    st.sidebar.title("Opcje Wyszukiwania Semantycznego")
    st.markdown("### Hello Semantic Search")

if __name__ == '__main__':
    app.run()
