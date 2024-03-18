import logging
import time
import streamlit as st
from itertools import groupby
from operator import itemgetter
from collocations import (compute_log_likelihood_scores_batch_partitioned,
                          filter_tokens_by_doc_ids, find_collocations,
                          find_relevant_doc_ids, find_detailed_collocation_occurrences,get_context_for_collocations)
from user_input_functions import get_user_input_collocations

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

def display_collocations():
    # Step 1: Get user inputs
    (selected_journals, year_range, query_lemma, search_type,
     left_context_length, right_context_length, n,
     is_stop, is_punct, selected_pos) = get_user_input_collocations()

    # Step 2: Check for button press to start search
    if st.sidebar.button('Wyszukaj kolokacje'):
        if query_lemma:
            with st.status("Wyszukiwanie danych tekstowych dla wybranych kryteriów",
                           expanded=False, state='running') as status:
                start_time = time.time()

                # Step 3: Find relevant documents based on user input
                relevant_doc_ids = find_relevant_doc_ids(selected_journals, year_range)
                filtered_df = filter_tokens_by_doc_ids(relevant_doc_ids)

                # Step 4: Search for collocations
                logger.info(f"Wyszukiwanie kolokacji dla segmentu '{query_lemma}' w podanym zakresie")
                start_time_collocations = time.time()
                collocations_result = find_collocations(filtered_df, query_lemma, search_type,
                                                        left_context_length, right_context_length,
                                                        is_stop, is_punct, selected_pos)
                time_for_collocations = time.time() - start_time_collocations
                logger.info(f"Znaleziono kolokacji: {len(collocations_result)}, czas wykonania: {time_for_collocations:.2f}s.")

                # Step 5: Compute log-likelihood scores for the collocations
                start_time_log_likelihood = time.time()
                status.update(label="Obliczanie log-likelihood. Analiza może potrwać od kilku do kilkunastu minut", expanded=True, state="running")
                total_tokens = len(filtered_df)
                log_likelihood_scores = compute_log_likelihood_scores_batch_partitioned(collocations_result, total_tokens, search_type)
                time_for_log_likelihood = time.time() - start_time_log_likelihood
                logger.info(f"Obliczanie log-likelihood zakończone. Czas wykonania: {time_for_log_likelihood:.2f}s.")

                # Step 6: Prepare the final data structure combining log-likelihood scores with doc_ids
                top_collocations_with_details = []
                for collocation, (count, doc_ids) in collocations_result.items():
                    score = log_likelihood_scores.get(collocation, 0)
                    top_collocations_with_details.append((collocation, score, doc_ids))
                top_collocations_with_details.sort(key=lambda item: item[1], reverse=True)
                top_collocations_with_details = top_collocations_with_details[:n]



                # Step 7: Display the strongest collocations to the user along with detailed occurrences
                status.update(label=f"Wyszukiwanie przykładów dla poszczególnych kolokacji", expanded=True, state="running")
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

                context_occurrences = get_context_for_collocations(detailed_occurrences, filtered_df)

                elapsed_time = time.time() - start_time
                status.update(label=f"Zakończenie analizy. Całkowity czas wykonania operacji: {elapsed_time:.2f}s.", state="complete", expanded=False)

            # Modyfikacja sposobu wyświetlania wyników
            st.write("Najsilniejsze kolokacje z kontekstem:")

            # Przygotowanie danych do grupowania
            context_occurrences.sort(key=itemgetter('collocation'))

            for key, group in groupby(context_occurrences, key=itemgetter('collocation')):
                # Zbieranie grupy do listy, aby móc wielokrotnie jej używać
                group_list = list(group)
                score = group_list[0]['score']  # Zakładamy, że score jest taki sam dla wszystkich kontekstów danej kolokacji
                expander_title = f"<b style='font-size: 18px;'>{key} - {score}</b>"  # Modyfikacja tytułu zgodnie z wymaganiami
                with st.expander(expander_title, expanded=False):
                    for context_data in group_list:
                        st.write(f"{context_data['context']} (Document ID: {context_data['doc_id']})")
                        st.write("")

display_collocations()
