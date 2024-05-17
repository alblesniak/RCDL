import logging
import os
import pickle
import sqlite3
import dask.dataframe as dd
import pandas as pd

import requests
import scipy.sparse
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resource_manager")

class ResourceManager:
    def __init__(self, chunksize=1024):
        self.chunksize = chunksize
        self.lemmas_to_idx = None
        self.tokens_to_idx = None
        self.idx_to_lemmas = None
        self.idx_to_tokens = None
        self.lemmas_sparse_matrix = None
        self.tokens_sparse_matrix = None
        self.idx_to_metadata = None
        self.tokens_df = None
        self.tokens_data = None

    def file_exists(self, filename):
        return os.path.exists(filename)

    def get_huggingface_token(self):
        hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        if not hf_token:
            load_dotenv()
            hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        return hf_token

    def download_file(self, url, filename, private_repo=False):
        if self.file_exists(filename):
            logger.info(f"File already exists: {filename}")
            return True

        headers = {}
        if private_repo:
            hf_token = self.get_huggingface_token()
            if not hf_token:
                raise ValueError("Hugging Face token is not set for private repo access.")
            headers = {"Authorization": f"Bearer {hf_token}"}

        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(filename, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            logger.info(f"File downloaded: {filename}")
            return True
        except Exception as e:
            logger.error(f"Error downloading file {filename}: {e}")
            return False

    def load_data(self):
        logger.info("Starting to load data")
        base_url = "https://huggingface.co/datasets/alblesniak/RCDL_dataset/resolve/main/"
        datasets_urls = {
            # Existing files
            "lemmas_to_idx.pkl": base_url + "lemmas_to_idx.pkl",
            "tokens_to_idx.pkl": base_url + "tokens_to_idx.pkl",
            "idx_to_lemmas.pkl": base_url + "idx_to_lemmas.pkl",
            "idx_to_tokens.pkl": base_url + "idx_to_tokens.pkl",
            "lemmas_sparse_matrix.npz": base_url + "lemmas_sparse_matrix.npz",
            "tokens_sparse_matrix.npz": base_url + "tokens_sparse_matrix.npz",
            "idx_to_metadata.parquet": base_url + "idx_to_metadata.parquet",
            # New files
            "Katolickie_per_year.csv": base_url + "Katolickie_per_year.csv",
            "Pentekostalne_per_year.csv": base_url + "Pentekostalne_per_year.csv",
            "keywords_data.xlsx": base_url + "keywords_data.xlsx",
        }

        private_datasets_urls = {
            "tokens_data.parquet": "https://huggingface.co/datasets/alblesniak/RCDL_tokens_data/resolve/main/tokens_data.parquet"
        }

        for name, url in {**datasets_urls, **private_datasets_urls}.items():
            file_path = f"data/{name}"
            attr_name = name.rsplit('.', 1)[0]  # Removing file extension
            if self.download_file(url, file_path, private_repo=name in private_datasets_urls):
                try:
                    if name.endswith('.pkl'):
                        with open(file_path, 'rb') as f:
                            setattr(self, attr_name, pickle.load(f))
                    elif name.endswith('.npz'):
                        setattr(self, attr_name, scipy.sparse.load_npz(file_path))
                    elif name.endswith('.parquet'):
                        if "tokens_data" in name:
                            self.tokens_data = self.load_parquet(file_path)
                        else:
                            setattr(self, attr_name, pd.read_parquet(file_path))
                    elif name.endswith('.csv'):
                        setattr(self, attr_name, pd.read_csv(file_path))
                    elif name.endswith('.xlsx'):
                        setattr(self, attr_name, pd.read_excel(file_path))
                    # Additional check for loaded data
                    if getattr(self, attr_name) is None:
                        logger.error(f"Data {attr_name} was not loaded correctly.")
                    else:
                        logger.info(f"Data {attr_name} loaded successfully.")
                except Exception as e:
                    logger.error(f"Error loading data {attr_name}: {e}")

    def load_parquet(self, file_path):
        logger.info(f"Start loading data from {file_path} using Dask.")
        try:
            dask_df = dd.read_parquet(file_path, engine='pyarrow', blocksize=self.chunksize)
            logger.info("Data loaded using Dask.")
            return dask_df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise e

def load_resources(resource_manager):
    """Loads resources needed for collocation analysis."""
    try:
        logger.info("Loading resources...")
        resource_manager.load_data()
        # Additional check if essential data has been loaded
        if resource_manager.lemmas_to_idx is None or resource_manager.tokens_to_idx is None:
            raise ValueError("Some resources were not loaded correctly.")
        logger.info('Resources loaded successfully.')
    except Exception as e:
        logger.error(f"Failed to load resources: {e}")
        raise

# Initialize database connection and cursor
def create_db_connection(db_path):
    """
    Establishes a database connection.

    :param db_path: Path to the database file.
    :return: A tuple of the connection and cursor objects.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    # Enable Write-Ahead Logging for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-20000")  # 20MB page-cache
    conn.execute("PRAGMA foreign_keys=ON")  # Enforce foreign-key constraints
    cur = conn.cursor()
    return conn, cur

# Example usage
if __name__ == "__main__":
    resource_manager = ResourceManager()
    load_resources(resource_manager)
