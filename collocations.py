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
    for doc_id, idx in stqdm(tokens_df[tokens_df[search_column] == query_lemma].index, desc="Trwa przetwrzanie..."):
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
    counts = np.array([count for count, doc_ids in collocations.values()])


    num_batches = int(np.ceil(len(lemmas) / batch_size))
    
    for i in stqdm(range(num_batches), desc="Trwa przetwrzanie..."):
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


def find_detailed_collocation_occurrences(top_collocations_with_details, tokens_df, query_lemma, search_type, left_context_length, right_context_length, is_stop, is_punct, selected_pos):
    detailed_collocations_occurrences = []

    # Przekształcenie tokens_df do słownika dla szybszego dostępu
    tokens_dict = {doc_id: df_group.to_dict(orient='records') for doc_id, df_group in tokens_df.groupby('doc_id')}

    search_column = 'lemma' if search_type == 'lemmas' else 'token_text'
    
    for collocation, score, doc_ids in top_collocations_with_details:
        collocation_examples = {}
        for doc_id in doc_ids:
            if doc_id not in tokens_dict:
                continue
            doc_tokens = tokens_dict[doc_id]

            node_indices = [i for i, token in enumerate(doc_tokens) if token[search_column] == query_lemma]
            collocation_indices = [i for i, token in enumerate(doc_tokens) if token[search_column] == collocation]

            for node_idx in node_indices:
                examples = []
                for collocation_idx in collocation_indices:
                    if abs(node_idx - collocation_idx) <= max(left_context_length, right_context_length):
                        context_row = doc_tokens[collocation_idx]
                        if filter_conditions(context_row, is_stop, is_punct, selected_pos):
                            examples.append((doc_tokens[node_idx]['id'], context_row['id']))
                if examples:
                    collocation_examples.setdefault(doc_id, []).extend(examples)

        if collocation_examples:
            # Dodanie wyniku score do listy
            detailed_collocations_occurrences.append([query_lemma, collocation, score, collocation_examples])

    return detailed_collocations_occurrences




def get_context_for_collocations(detailed_collocations_occurrences, tokens_df, left_context_length=5, right_context_length=5):
    context_cache = {}
    context_results = []

    tokens_dict = {doc_id: df_group.to_dict(orient='records') for doc_id, df_group in tokens_df.groupby('doc_id')}

    for item in detailed_collocations_occurrences:
        query_lemma, collocation, score, doc_ids_dict = item  # Zmienione rozpakowanie zmiennych, dodano score
        for doc_id, ids_pairs in doc_ids_dict.items():
            if doc_id not in tokens_dict:
                continue
            for pair in ids_pairs:
                cache_key = (doc_id,) + pair
                if cache_key in context_cache:
                    context_str = context_cache[cache_key]
                else:
                    node_id, collocate_id = pair
                    context_start = max(1, min(node_id, collocate_id) - left_context_length)
                    context_end = max(node_id, collocate_id) + right_context_length
                    context_tokens = [token['token_text'] for token in tokens_dict[doc_id] if context_start <= token['id'] <= context_end]
                    context_str = ' '.join(context_tokens)
                    context_cache[cache_key] = context_str

                context_results.append({
                    'query_lemma': query_lemma,
                    'collocation': collocation,
                    'score': score,  # Używanie zmiennej score
                    'doc_id': doc_id,
                    'context': context_str
                })

    return context_results


