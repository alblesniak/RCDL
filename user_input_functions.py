import streamlit as st


def get_selected_journal_and_year_range(issue_info_dict):
    """Get user-selected journal and year range from the sidebar using issue_info_dict."""

    # Get unique journal names from the issue_info_dict
    journal_names = list(issue_info_dict.keys())

    # Create a multiselect field for journal selection
    selected_journals = st.sidebar.multiselect(
        'Wybierz nazwy wydań:', journal_names, placeholder="Wybierz opcję")

    # If journals are selected
    if selected_journals:
        # Initialize min_year and max_year with the first selected journal's values
        min_year = issue_info_dict[selected_journals[0]]['min_year']
        max_year = issue_info_dict[selected_journals[0]]['max_year']

        # Iterate through selected journals to find the overall min_year and max_year
        for journal in selected_journals:
            journal_info = issue_info_dict[journal]
            min_year = min(min_year, journal_info['min_year'])
            max_year = max(max_year, journal_info['max_year'])

        # Create a year range slider based on the selected journals' min and max years
        selected_year_range = st.sidebar.slider(
            'Wybierz zakres lat:', min_year, max_year, (min_year, max_year))

        return selected_journals, selected_year_range

    else:
        return None, None



def get_query_lemma():
    """Get user input for query lemma."""
    return st.sidebar.text_input('Wprowadź segment węzłowy:')


def get_search_type():
    """Get user-selected search type."""
    search_type_options = {
        'Lematy (formy bazowe)': 'lemmas', 'Tokeny (formy gramatyczne)': 'tokens'}
    search_type_label = st.sidebar.radio(
        'Wybierz typ wyszukiwania:', list(search_type_options.keys()))
    return search_type_options[search_type_label]


def get_context_length():
    """Get user-selected context length."""
    col1, col2 = st.sidebar.columns(2)
    with col1:
        left_context_length = st.number_input(
            'Liczba słów lewego kontekstu:', 1, 25, 5)
    with col2:
        right_context_length = st.number_input(
            'Liczba słów prawego kontekstu:', 1, 25, 5)
    return left_context_length, right_context_length


def get_collocation_params():
    """Get user-selected parameters for collocations."""
    n = st.sidebar.number_input('Liczba szukanych kolokatów:', 1, 100, 15)
    is_stop = st.sidebar.checkbox('Uwzględniaj "stopwordy"', False)
    is_punct = st.sidebar.checkbox('Uwzględniaj interpunkcję', False)
    return n, is_stop, is_punct


def get_part_of_speech():
    """Get user-selected part of speech filter."""
    pos_mapping = {
        'Przymiotniki': 'ADJ',
        'Rzeczowniki': 'NOUN',
        'Czasowniki': 'VERB',
        'Przyimki': 'ADP',
        'Przysłówki': 'ADV',
        'Interpunkcja': 'PUNCT',
        'Spójniki': 'CCONJ',
        'Cząstki': 'PART',
        'Zaimki': 'PRON',
        'Spójniki podrzędne': 'SCONJ',
        'Liczbniki': 'NUM',
        'Czasowniki pomocnicze': 'AUX',
        'Wykrzykniki': 'INTJ'
    }
    selected_pos_label = st.sidebar.selectbox(
        'Ogranicz do części mowy', [''] + list(pos_mapping.keys()))
    return pos_mapping.get(selected_pos_label, None)


def get_user_input_collocations():
    """Collect all user inputs related to collocations."""
    issue_info_dict = {
        'Przewodnik Katolicki': {'min_year': 2008, 'max_year': 2023},
        'Niedziela': {'min_year': 2010, 'max_year': 2023},
        'Gość Niedzielny': {'min_year': 2005, 'max_year': 2023},
        'Chrześcijanin': {'min_year': 1989, 'max_year': 2022}
    }
    selected_journals, year_range = get_selected_journal_and_year_range(
        issue_info_dict)
    query_lemma = get_query_lemma()
    search_type = get_search_type()
    left_context_length, right_context_length = get_context_length()
    n, is_stop, is_punct = get_collocation_params()
    selected_pos = get_part_of_speech()
    return selected_journals, year_range, query_lemma, search_type, left_context_length, right_context_length, n, is_stop, is_punct, selected_pos
