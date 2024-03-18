import logging
import time

import psutil
import streamlit as st

from collocations import (compute_log_likelihood_scores_batch_partitioned,
                          filter_tokens_by_doc_ids, find_collocations,
                          find_relevant_doc_ids,find_detailed_collocation_occurrences)
from user_input_functions import get_user_input_collocations

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")



def display_collocations():
    # Step 1: Get user inputs
    # Obtain input from the user such as selected journals, year range, query lemma, etc.
    (selected_journals, year_range, query_lemma, search_type, 
     left_context_length, right_context_length, n, 
     is_stop, is_punct, selected_pos) = get_user_input_collocations()
    
    # Step 2: Check for button press to start search
    # When the user clicks the 'Search collocations' button in the Streamlit sidebar
    if st.sidebar.button('Wyszukaj kolokacje'):
        # Ensure a query lemma is provided to proceed
        if query_lemma:
            # Display a status message while the operation is running
            with st.status("Wyszukiwanie danych tekstowych dla wybranych kryteriów", 
                           expanded=False, state='running') as status:

                # Record the start time of the operation
                start_time = time.time()

                # Step 3: Find relevant documents based on user input
                relevant_doc_ids = find_relevant_doc_ids(selected_journals, year_range)
                filtered_df = filter_tokens_by_doc_ids(relevant_doc_ids)
                
                # Step 4: Search for collocations
                # Log the start of collocation search
                logger.info(f"Wyszukiwanie kolokacji dla segmentu '{query_lemma}' w podanym zakresie")
                start_time_collocations = time.time()
                
                # Find collocations based on the filtered DataFrame and user inputs
                collocations_result = find_collocations(filtered_df, query_lemma, search_type, 
                                                        left_context_length, right_context_length, 
                                                        is_stop, is_punct, selected_pos)
                time_for_collocations = time.time() - start_time_collocations
                # Update the status with the number of collocations found and the execution time
                logger.info(f"Znaleziono kolokacji: {len(collocations_result)}, czas wykonania: {time_for_collocations:.2f}s.")
                
                # Step 5: Compute log-likelihood scores for the collocations
                start_time_log_likelihood = time.time()
                status.update(label=f"Obliczanie log-likelihood. Analiza może potrwać od kilku do kilkunastu minut", expanded=True, state="running")
                total_tokens = len(filtered_df)
                log_likelihood_scores = compute_log_likelihood_scores_batch_partitioned(collocations_result, total_tokens, search_type)
                time_for_log_likelihood = time.time() - start_time_log_likelihood
                logger.info(f"Obliczanie log-likelihood zakończone. Czas wykonania: {time_for_log_likelihood:.2f}s.")
                

                # Step 6: Prepare the final data structure combining log-likelihood scores with doc_ids
                top_collocations_with_details = []
                for collocation, (count, doc_ids) in collocations_result.items():
                    score = log_likelihood_scores.get(collocation, 0)  # Get log-likelihood score, defaulting to 0 if not found
                    top_collocations_with_details.append((collocation, score, doc_ids))
                
                # Sort the combined data based on log-likelihood score and trim to `n` elements
                top_collocations_with_details.sort(key=lambda item: item[1], reverse=True)
                top_collocations_with_details = top_collocations_with_details[:n]


                # Step 7: Display the strongest collocations to the user along with detailed occurrences
                status.update(label=f"Wyszukiwanie przykładów dla poszczególnych kolokacji", expanded=True, state="running")
                detailed_occurrences = find_detailed_collocation_occurrences(
                    top_collocations_with_details=top_collocations_with_details,
                    tokens_df=filtered_df,
                    query_lemma=query_lemma,  # Upewnij się, że jest to poprawnie przekazane
                    search_type=search_type,
                    left_context_length=left_context_length,
                    right_context_length=right_context_length,
                    is_stop=is_stop,
                    is_punct=is_punct,
                    selected_pos=selected_pos  # Upewnij się, że ten argument jest poprawnie przekazany
                )
            
            # Wyświetlenie wyników
            st.write("Najsilniejsze kolokacje z identyfikatorami wystąpień:")
            st.json(detailed_occurrences)

            elapsed_time = time.time() - start_time
            status.update(label=f"Zakończenie analizy. Całkowity czas wykonania operacji: {elapsed_time:.2f}s.", state="complete", expanded=False)

display_collocations()