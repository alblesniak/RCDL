import pandas as pd

def load_keywords_data(corpus_choice):
    file_path = './data/keywords_data.xlsx'
    time_series_file_path = f'./data/{corpus_choice}_per_year.csv'

    df = pd.read_excel(file_path, sheet_name=corpus_choice)
    time_series_df = pd.read_csv(time_series_file_path)

    # Połącz główny dataframe z danymi czasowymi
    merged_df = df.merge(time_series_df, on='keyword', how='left')

    # Wygeneruj listę 'occurrences_over_time' dla każdego wiersza
    years = time_series_df.columns.difference(['keyword']).tolist()

    # Konwertuj kolumny z latami na typ float
    for year in years:
        merged_df[year] = merged_df[year].astype(float)

    merged_df['occurrences_over_time'] = merged_df[years].values.tolist()

    # Usuń niepotrzebne kolumny związane z latami, ale zachowaj kolumnę 'keyword'
    merged_df.drop(columns=years, inplace=True)

    return merged_df
