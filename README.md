# Research Paper Analyzer

A Python tool for analyzing and tracking (industrial visual inspection) research papers using the OpenAI API.
The project is inspired by Wesley Pasfield's paper newsletter (https://github.com/WesleyPasfield/paper-newsletter).

## Setup

1. Install the conda environment using the provided environment.yml file
    ```bash
    conda env create -f environment.yml
    ```
2. Create a .env file in the root directory and add the following variables:
    - OPENAI_API_KEY: Your OpenAI API key

## Usage

1. Run the analyzer script
    ```bash
    python src/analyzer.py
    ```
2. Run the streamlit app (run it with the `--server.address 0.0.0.0` option to access the app from other devices)
    ```bash
    streamlit run src/viewer.py
    ```
3. Open the streamlit app in your browser (default: http://localhost:8501)