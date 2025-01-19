# Research Paper Analyzer

A Python tool for analyzing and tracking (industrial visual inspection) research papers using the OpenAI API.
The project is inspired by Wesley Pasfield's paper newsletter (https://github.com/WesleyPasfield/paper-newsletter).

## Setup

1. Install the conda environment using the provided `environment.yml` file
    ```bash
    conda env create -f environment.yml
    ```
2. Create a `.env` file in the root directory and add the following variables:
    - `OPENAI_API_KEY`: Your OpenAI API key

## Usage

### Local

1. Run the analyzer script
    ```bash
    python src/analyzer.py
    ```
2. Run the streamlit app (run it with the `--server.address 0.0.0.0` option to access the app from other devices)
    ```bash
    streamlit run src/viewer.py
    ```
3. Open the streamlit app in your browser (default: http://localhost:8501)

### GitHub Actions

If both the `GITHUB_REPOSITORY` and the `GITHUB_TOKEN` environment variables are set, the analyzer script will save the
output as GitHub release artifacts (instead of saving them locally). The `.github/workflows/daily-analysis.yml` file
specifies a GitHub Actions workflow that runs the analyzer script daily at 6:00 AM UTC. The environment variables
`GITHUB_TOKEN` and `OPENAI_API_KEY` must be provided as secrets in this case. 

### Streamlit Community App

The streamlit app can be deployed using the Streamlit Sharing platform with only a few clicks. Note that the 
`GITHUB_REPOSITORY` and `GITHUB_TOKEN` environment variables must be set so that the app can access the GitHub releases.

To get an impression of how the app looks like, you can visit the following link: https://research-analyzer-thzpdzxdfjugah4dprxspg.streamlit.app/

## Possible Extensions

Both the `analyzer.py` and the `viewer.py` scripts can be extended in various ways. Some ideas are:
- Use a different source as input, e.g. other arXiv feeds, Hacker News, etc.
- Change the prompts to scan for content of a different application domain.
- Make the streamlit app more interactive, easier to use for your specific use case, etc.

In addition, it is possible to change the way the data is stored and accessed, i.e. move away from GitHub releases and
use better suited storage solution instead.