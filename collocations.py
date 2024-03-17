import pandas as pd
import logging
import math
from collections import Counter
import dask.dataframe as dd
import numpy as np
import scipy.sparse
from tqdm import tqdm
import psutil
from resource_manager import ResourceManager
from stqdm import stqdm

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("collocations")

# Inicjalizacja ResourceManager
resource_manager = ResourceManager()

def load_resources():
    """Ładuje zasoby potrzebne do analizy kolokacji."""
    try:
        logger.info("Loading resources...")
        resource_manager.load_data()
        # Dodatkowe sprawdzenie, czy kluczowe dane zostały załadowane
        if resource_manager.lemmas_to_idx is None or resource_manager.tokens_to_idx is None:
            raise ValueError("Niektóre zasoby nie zostały załadowane poprawnie.")
        logger.info('Resources loaded successfully.')
    except Exception as e:
        logger.error(f"Failed to load resources: {e}")
        raise

# Wywołanie funkcji ładowania zasobów przy starcie aplikacji
load_resources()

def find_relevant_doc_ids(selected_journals, year_range):
    """Wyszukuje doc_id na podstawie selected_journals i year_range."""
    if resource_manager.idx_to_metadata is None:
        logger.error("Metadata index dictionary is not loaded.")
        return set()

    metadata = resource_manager.idx_to_metadata
    filtered_metadata = metadata[(metadata['issue_name'].isin(selected_journals)) & (metadata['year'].between(year_range[0], year_range[1]))]
    logger.info(f'Found {len(filtered_metadata)} documents matching criteria.')
    relevant_doc_ids = set(filtered_metadata.index)
    return relevant_doc_ids


def filter_tokens_by_doc_ids(relevant_doc_ids):
    """
    Filters tokens_df based on found doc_ids using Dask.
    """
    if resource_manager.tokens_data is None:
        logger.error("Tokens data is not loaded.")
        return dd.from_pandas(pd.DataFrame(), npartitions=1)  # Zwraca pusty Dask DataFrame

    tokens_dask_df = resource_manager.tokens_data
    
    # Przekonwertuj multi-indeks na kolumny przed filtrowaniem
    tokens_dask_df = tokens_dask_df.reset_index() 

    # Filtering directly without converting to pandas DataFrame
    filtered_dask_df = tokens_dask_df[tokens_dask_df['doc_id'].isin(relevant_doc_ids)]

    # Konwersja na Pandas
    tokens_df = filtered_dask_df.compute()
    
    logger.info(f'Filtered tokens dataframe size: {tokens_df.shape[0]}')
    return tokens_df


def filter_conditions(row, is_stop, is_punct, selected_pos):
    """Determines if a row meets the specified filtering conditions."""
    pos_condition = True if selected_pos is None else row['pos'] == selected_pos
    return (is_stop or not row['is_stop']) and (is_punct or not row['is_punct']) and pos_condition

def get_context_indices(doc_id, idx, tokens_df, left_context_length, right_context_length):
    """Calculates the range of indices for the context around a target node."""
    return range(max(1, idx - left_context_length), min(tokens_df.loc[doc_id].index.max(), idx + right_context_length) + 1)

def find_collocations(tokens_df, query_lemma, search_type, left_context_length, right_context_length, is_stop, is_punct, selected_pos):
    """Identifies collocations for the specified node segment with various filtering options."""
    # Sprawdzenie, czy kluczowe zasoby zostały załadowane
    if search_type == 'lemmas' and resource_manager.lemmas_to_idx is None:
        logger.error("Lemmas index dictionary is not loaded.")
        return {}
    elif search_type == 'tokens' and resource_manager.tokens_to_idx is None:
        logger.error("Tokens index dictionary is not loaded.")
        return {}

    search_column = 'lemma' if search_type == 'lemmas' else 'token_text'
    collocations = {}  # Zmodyfikowana struktura danych
    target_idx = resource_manager.lemmas_to_idx.get(query_lemma) if search_type == 'lemmas' else resource_manager.tokens_to_idx.get(query_lemma)

    if target_idx is None:
        logger.error(f"Node segment '{query_lemma}' not found in index.")
        return {}

    tokens_df = tokens_df.set_index(['doc_id', 'id'])
    for doc_id, idx in tokens_df[tokens_df[search_column] == query_lemma].index:
        context_indices = get_context_indices(doc_id, idx, tokens_df, left_context_length, right_context_length)
        for context_idx in context_indices:
            if context_idx != idx and (doc_id, context_idx) in tokens_df.index:
                context_row = tokens_df.loc[(doc_id, context_idx)]
                if filter_conditions(context_row, is_stop, is_punct, selected_pos):
                    context_lemma = context_row[search_column]
                    if context_lemma not in collocations:
                        collocations[context_lemma] = (0, set())
                    count, doc_ids = collocations[context_lemma]
                    collocations[context_lemma] = (count + 1, doc_ids | {doc_id})

    # Konwersja zestawów doc_id na listy dla lepszej serializacji/eksportu
    for key, (count, doc_ids) in collocations.items():
        collocations[key] = (count, list(doc_ids))

    return collocations



def calculate_log_likelihood_batch(O11, O12, O21, O22):
    """Computes the log likelihood score for a batch of collocations."""
    total = O11 + O12 + O21 + O22
    E11 = np.where(total > 0, (O11 + O12) * (O11 + O21) / total, 0)
    E12 = np.where(total > 0, (O11 + O12) * (O12 + O22) / total, 0)
    E21 = np.where(total > 0, (O21 + O22) * (O11 + O21) / total, 0)
    E22 = np.where(total > 0, (O21 + O22) * (O12 + O22) / total, 0)

    log_likelihood = np.where((E11 > 0) & (O11 > 0), O11 * np.log(np.where(O11 / E11 > 0, O11 / E11, 1)), 0) + \
                     np.where((E12 > 0) & (O12 > 0), O12 * np.log(np.where(O12 / E12 > 0, O12 / E12, 1)), 0) + \
                     np.where((E21 > 0) & (O21 > 0), O21 * np.log(np.where(O21 / E21 > 0, O21 / E21, 1)), 0) + \
                     np.where((E22 > 0) & (O22 > 0), O22 * np.log(np.where(O22 / E22 > 0, O22 / E22, 1)), 0)
                     
    return 2 * log_likelihood

def compute_log_likelihood_scores_batch_partitioned(collocations, total_tokens, search_type, batch_size=10):
    """Computes log likelihood scores for each collocation in batches with partitioning, based on search_type."""
    # Sprawdzenie, czy zasoby dla lematów lub tokenów zostały poprawnie załadowane
    if search_type == 'lemmas' and (resource_manager.lemmas_sparse_matrix is None or resource_manager.lemmas_to_idx is None):
        logger.error("Lemmas resources are not loaded.")
        return {}
    elif search_type == 'tokens' and (resource_manager.tokens_sparse_matrix is None or resource_manager.tokens_to_idx is None):
        logger.error("Tokens resources are not loaded.")
        return {}

    scores_dict = {}
    # Wybór odpowiedniej macierzy rzadkiej i indeksu na podstawie search_type
    sparse_matrix = resource_manager.lemmas_sparse_matrix if search_type == 'lemmas' else resource_manager.tokens_sparse_matrix
    column_to_idx = resource_manager.lemmas_to_idx if search_type == 'lemmas' else resource_manager.tokens_to_idx

    lemmas = np.array(list(collocations.keys()))
    counts = np.array(list(collocations.values()))

    num_batches = int(np.ceil(len(lemmas) / batch_size))
    
    for i in stqdm(range(num_batches), desc="Processing..."):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(lemmas))
        lemmas_batch = lemmas[start_index:end_index]
        counts_batch = counts[start_index:end_index]

        lemma_indices = np.array([column_to_idx.get(lemma) for lemma in lemmas_batch if lemma in column_to_idx])

        # Kontynuuj tylko, jeśli wszystkie indeksy są dostępne
        if not all(lemma_indices):
            continue

        O11 = counts_batch
        O12 = np.array([sparse_matrix[:, idx].sum() if idx is not None else 0 for idx in lemma_indices]) - counts_batch
        O21 = counts_batch.sum() - counts_batch
        O22 = total_tokens - O11 - O12 - O21

        batch_scores = calculate_log_likelihood_batch(O11, O12, O21, O22)
        for lemma, score in zip(lemmas_batch, batch_scores):
            scores_dict[lemma] = score

    return scores_dict
