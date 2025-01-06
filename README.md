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