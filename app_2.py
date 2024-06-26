import logging
import time
import uuid
import hydralit as hy
import streamlit as st
from itertools import groupby
from keywords import load_keywords_data
from collocations import (compute_log_likelihood_scores_batch_partitioned,
                          filter_tokens_by_doc_ids, find_collocations,
                          find_relevant_doc_ids, find_detailed_collocation_occurrences, get_context_for_collocations, generate_markdown_report)
from user_input_functions import get_user_input_collocations
from annotated_text import annotated_text

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")


# Hydralit app setup
app = hy.HydraApp(
    title='Religious Cultures DataLab',
    sidebar_state='expanded',
    use_loader=False,
    hide_streamlit_markers=True,
    use_navbar=True,
    navbar_sticky=True,
    navbar_animation=False,
)


# @app.addapp(title='Słowa kluczowe', is_home=True,)
# def keywords():
#     """Page for handling keywords."""
#     hy.info('Witaj w sekcji słów kluczowych!')


# Moduł Słowa Kluczowe
@app.addapp(title='Słowa kluczowe', is_home=True)
def keywords():
    # Callback to reset pagination
    def on_corpus_change():
        st.session_state.current_page = 0
    # Initialize session state for pagination and corpus selection
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 0

    # User Selection for 'corpus' filtering in sidebar
    options = ['Katolickie', 'Pentekostalne']
    selected_corpus = st.sidebar.selectbox(
        'Wybierz korpus:', options, key='selected_corpus', on_change=on_corpus_change)

    # Load and process data based on the selection
    merged_df = load_keywords_data(selected_corpus)

    # Sort dataframe by 'log_likelihood' - descending
    merged_df = merged_df.sort_values(by='log_likelihood', ascending=False)

    # Pagination logic
    items_per_page = 100
    num_pages = len(merged_df) // items_per_page + (1 if len(merged_df) % items_per_page > 0 else 0)

    # Display Dataframe with custom column configurations
    st.dataframe(
        merged_df.iloc[st.session_state.current_page * items_per_page:(st.session_state.current_page + 1) * items_per_page],
        column_config={
            "keyword": st.column_config.TextColumn("Słowo kluczowe", width="medium"),
            "log_likelihood": st.column_config.NumberColumn("Log Likelihood"),
            "occurrences_A": st.column_config.NumberColumn("Wystąpienia w korpusie K"),
            "occurrences_per_1000_A": st.column_config.NumberColumn("Wystąpienia na 1000 słów (K)"),
            "occurrences_B": st.column_config.NumberColumn("Wystąpienia w korpusie P"),
            "occurrences_per_1000_B": st.column_config.NumberColumn("Wystąpienia na 1000 słów (P)"),
            "occurrences_over_time": st.column_config.LineChartColumn("Wystąpienia w czasie"),
        },
        hide_index=True,
        width=1500,
        height=600
    )

    # Pagination controls
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
    with col2:
        if st.button('Poprzednia strona'):
            if st.session_state.current_page > 0:
                st.session_state.current_page -= 1
                st.experimental_rerun()

    with col3:
        st.text(f"Strona {st.session_state.current_page + 1} z {num_pages}")

    with col4:
        if st.button('Następna strona'):
            if st.session_state.current_page < num_pages - 1:
                st.session_state.current_page += 1
                st.experimental_rerun()


@app.addapp(title='Kolokacje')
def collocations():
    # Step 1: Initialize session state variables if they don't exist
    if 'context_occurrences' not in st.session_state:
        st.session_state.context_occurrences = []

    # Get user inputs
    user_inputs = get_user_input_collocations()

    # Step 2: Check for button press to start search
    if st.sidebar.button('Wyszukaj kolokacje'):
        (selected_journals, year_range, query_lemma, search_type,
         left_context_length, right_context_length, n,
         is_stop, is_punct, selected_pos) = user_inputs
        if query_lemma:
            with st.spinner("Wyszukiwanie danych tekstowych dla wybranych kryteriów"):
                start_time = time.time()

                # Find relevant documents based on user input
                relevant_doc_ids = find_relevant_doc_ids(selected_journals, year_range)
                filtered_df = filter_tokens_by_doc_ids(relevant_doc_ids)

                # Search for collocations
                logger.info(f"Wyszukiwanie kolokacji dla segmentu '{query_lemma}' w podanym zakresie")
                start_time_collocations = time.time()
                collocations_result = find_collocations(filtered_df, query_lemma, search_type,
                                                        left_context_length, right_context_length,
                                                        is_stop, is_punct, selected_pos)
                time_for_collocations = time.time() - start_time_collocations
                logger.info(f"Znaleziono kolokacji: {len(collocations_result)}, czas wykonania: {time_for_collocations:.2f}s.")

                # Compute log-likelihood scores for the collocations
                total_tokens = len(filtered_df)
                log_likelihood_scores = compute_log_likelihood_scores_batch_partitioned(collocations_result, total_tokens, search_type)

                # Prepare the final data structure combining log-likelihood scores with doc_ids
                top_collocations_with_details = []
                for collocation, (count, doc_ids) in collocations_result.items():
                    score = log_likelihood_scores.get(collocation, 0)
                    top_collocations_with_details.append((collocation, score, doc_ids))
                top_collocations_with_details.sort(key=lambda item: item[1], reverse=True)
                top_collocations_with_details = top_collocations_with_details[:n]

                # Display the strongest collocations to the user along with detailed occurrences
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

    # Display results
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

    # Download report if there are results
    if st.session_state.context_occurrences:
        markdown_content = generate_markdown_report(user_inputs, st.session_state.context_occurrences)
        unique_key = str(uuid.uuid4())
        st.download_button(label="Download Report as Markdown",
                           data=markdown_content,
                           file_name="collocation_report.md",
                           mime="text/markdown",
                           key=unique_key)

@app.addapp(title='Wektory słów')
def word_vectors():
    """Page for handling word vectors."""
    hy.info('Witaj w sekcji wektorów słów!')


@app.addapp(title='Modele tematyczne')
def topic_models():
    """Page for handling topic models."""
    hy.info('Witaj w sekcji modeli tematycznych!')


@app.addapp(title='Wyszukiwanie semantyczne')
def semantic_search():
    """Page for handling semantic search."""
    hy.info('Witaj w sekcji wyszukiwania semantycznego!')


# Running the app
app.run()