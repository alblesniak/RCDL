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

# @cache_data
def load_resources():
    """Ładuje zasoby potrzebne do analizy kolokacji."""
    logger.info("Loading resources...")
    # Additional logging to confirm resource load
    logger.info('Resources loaded successfully.')
    resource_manager.load_data()

# # Wywołanie funkcji ładowania zasobów przy starcie aplikacji
load_resources()


def find_relevant_doc_ids(selected_journals, year_range):
    """Wyszukuje doc_id na podstawie selected_journals i year_range."""
    metadata = resource_manager.idx_to_metadata
    filtered_metadata = metadata[(metadata['issue_name'].isin(selected_journals)) & (metadata['year'].between(year_range[0], year_range[1]))]
    logger.info(f'Found {len(filtered_metadata)} documents matching criteria.')
    relevant_doc_ids = set(filtered_metadata.index)
    return relevant_doc_ids


def filter_tokens_by_doc_ids(relevant_doc_ids):
    """
    Filters tokens_df based on found doc_ids using Dask.
    """
    
    tokens_dask_df = resource_manager.tokens_data
    
    # Ensure we are working with a Dask DataFrame
    if not isinstance(tokens_dask_df, dd.DataFrame):
        raise ValueError("Expected Dask DataFrame as tokens_dask_df")

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
    search_column = 'lemma' if search_type == 'lemmas' else 'token_text'
    collocations = Counter()
    target_idx = resource_manager.lemmas_to_idx.get(query_lemma)

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
                    collocations[context_lemma] += 1

    return collocations



def calculate_log_likelihood(O11, O12, O21, O22):
    """Computes the log likelihood score for a collocation."""
    total = O11 + O12 + O21 + O22
    E11 = (O11 + O12) * (O11 + O21) / total if total > 0 else 0
    E12 = (O11 + O12) * (O12 + O22) / total if total > 0 else 0
    E21 = (O21 + O22) * (O11 + O21) / total if total > 0 else 0
    E22 = (O21 + O22) * (O12 + O22) / total if total > 0 else 0
    log_likelihood = 0
    for O, E in [(O11, E11), (O12, E12), (O21, E21), (O22, E22)]:  # noqa: E741
        if E > 0 and O > 0:
            log_likelihood += O * math.log(O / E)
    return 2 * log_likelihood

def compute_log_likelihood_scores(collocations, total_tokens):
    """Computes log likelihood scores for each collocation."""
    lemmas_sparse_matrix = resource_manager.lemmas_sparse_matrix
    lemmas_to_idx = resource_manager.lemmas_to_idx

    log_likelihood_scores = {}
    for lemma, count in tqdm(collocations.items()):
        if lemma in lemmas_to_idx:
            target_idx = lemmas_to_idx[lemma]
            O11 = count
            O12 = lemmas_sparse_matrix[:, target_idx].sum() - count
            O21 = sum(collocations.values()) - count
            O22 = total_tokens - O11 - O12 - O21
            score = calculate_log_likelihood(O11, O12, O21, O22)
            log_likelihood_scores[lemma] = score

    logger.info(f'Computed log-likelihood scores for {len(log_likelihood_scores)} collocations.')
    return log_likelihood_scores

def calculate_log_likelihood_batch(O11, O12, O21, O22):
    """Computes the log likelihood score for a batch of collocations."""
    total = O11 + O12 + O21 + O22
    E11 = np.where(total > 0, (O11 + O12) * (O11 + O21) / total, 0)
    E12 = np.where(total > 0, (O11 + O12) * (O12 + O22) / total, 0)
    E21 = np.where(total > 0, (O21 + O22) * (O11 + O21) / total, 0)
    E22 = np.where(total > 0, (O21 + O22) * (O12 + O22) / total, 0)

    # Calculate log likelihood, ensuring we avoid log(0) by ensuring both O and E are greater than 0.
    log_likelihood = np.where((E11 > 0) & (O11 > 0), O11 * np.log(np.where(O11 / E11 > 0, O11 / E11, 1)), 0) + \
                     np.where((E12 > 0) & (O12 > 0), O12 * np.log(np.where(O12 / E12 > 0, O12 / E12, 1)), 0) + \
                     np.where((E21 > 0) & (O21 > 0), O21 * np.log(np.where(O21 / E21 > 0, O21 / E21, 1)), 0) + \
                     np.where((E22 > 0) & (O22 > 0), O22 * np.log(np.where(O22 / E22 > 0, O22 / E22, 1)), 0)
                     
    return 2 * log_likelihood


def compute_log_likelihood_scores_batch(collocations, total_tokens):
    """Computes log likelihood scores for each collocation in batches."""
    lemmas = np.array(list(collocations.keys()))
    counts = np.array(list(collocations.values()))

    lemmas_sparse_matrix = resource_manager.lemmas_sparse_matrix
    lemmas_to_idx = resource_manager.lemmas_to_idx

    # Convert sparse matrix to CSR format for efficient row/column slicing
    if not isinstance(lemmas_sparse_matrix, scipy.sparse.csr_matrix):
        lemmas_sparse_matrix = lemmas_sparse_matrix.tocsr()

    # Get indices for lemmas
    lemma_indices = np.array([lemmas_to_idx[lemma] for lemma in lemmas if lemma in lemmas_to_idx])
    
    # Compute O12 using correct sparse matrix indexing/slicing
    # Note: We need to ensure we're extracting columns efficiently
    O11 = counts
    O12 = np.array([lemmas_sparse_matrix[:, idx].sum() for idx in lemma_indices]) - counts
    O21 = counts.sum() - counts
    O22 = total_tokens - O11 - O12 - O21

    # Calculate log likelihood scores in a batch
    scores = calculate_log_likelihood_batch(O11, O12, O21, O22)

    # Map scores back to lemmas
    log_likelihood_scores = {lemma: score for lemma, score in zip(lemmas, scores)}

    return log_likelihood_scores

# @numba.jit(nopython=True)
def compute_log_likelihood_scores_batch_partitioned(collocations, total_tokens, batch_size=10):
    """Computes log likelihood scores for each collocation in batches with partitioning."""
    scores_dict = {}

    lemmas = np.array(list(collocations.keys()))
    counts = np.array(list(collocations.values()))
    lemmas_sparse_matrix = resource_manager.lemmas_sparse_matrix
    lemmas_to_idx = resource_manager.lemmas_to_idx

    num_batches = int(np.ceil(len(lemmas) / batch_size))
    
    for i in stqdm(range(num_batches), desc="Trwa przetwarzanie..."):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(lemmas))
        lemmas_batch = lemmas[start_index:end_index]
        counts_batch = counts[start_index:end_index]

        lemma_indices = np.array([lemmas_to_idx[lemma] for lemma in lemmas_batch if lemma in lemmas_to_idx])
        O11 = counts_batch
        O12 = np.array([lemmas_sparse_matrix[:, idx].sum() for idx in lemma_indices]) - counts_batch
        O21 = counts_batch.sum() - counts_batch
        O22 = total_tokens - O11 - O12 - O21

        batch_scores = calculate_log_likelihood_batch(O11, O12, O21, O22)
        for lemma, score in zip(lemmas_batch, batch_scores):
            scores_dict[lemma] = score

        # Logowanie zużycia procesora i pamięci RAM
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        logger.info(f"Aktualne zużycie CPU: {cpu_usage}%, RAM: {ram_usage}%")


    return scores_dict
