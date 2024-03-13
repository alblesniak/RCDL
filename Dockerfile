# Użyj oficjalnego obrazu Pythona jako obrazu bazowego.
# Wybierz wersję Pythona, która jest kompatybilna z Twoją aplikacją.
FROM python:3.11

# Ustaw /app jako katalog roboczy w kontenerze.
WORKDIR /app

# Zainstaluj narzędzia kompilacji i zależności systemowe niezbędne dla niektórych pakietów Python.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Skopiuj plik requirements.txt do katalogu roboczego.
COPY requirements.txt /app/

# Zainstaluj wszelkie potrzebne pakiety określone w requirements.txt.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Skopiuj resztę kodu źródłowego aplikacji do katalogu roboczego w kontenerze.
COPY . /app

# Ustaw zmienną środowiskową wskazującą, że aplikacja działa w kontenerze.
# To może być użyteczne, jeśli Twoja aplikacja potrzebuje specjalnego zachowania uruchamiana w kontenerze.
ENV RUNNING_IN_DOCKER=true

# Ustaw zmienną środowiskową Streamlit, aby uruchomić aplikację w trybie bezgłowym.
ENV STREAMLIT_SERVER_HEADLESS=true

# Ustaw zmienną środowiskową wskazującą port, na którym będzie działać Streamlit.
# Domyślnie Streamlit używa portu 8501, ale możesz to zmienić, jeśli jest taka potrzeba.
ENV STREAMLIT_SERVER_PORT=8501

# Informuj Docker, że kontener będzie słuchał na określonym porcie w czasie działania.
EXPOSE 8501

# Uruchom aplikację Streamlit przy starcie kontenera.
CMD ["streamlit", "run", "app.py"]
