import json
import os
import pandas as pd
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv
from pathlib import Path
from typing import Dict

from storage_utils import StorageManager


def load_analysis_results(file_path: str) -> Dict:
    """Load analysis results from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def render_papers_list(df: pd.DataFrame, category: str):
    """Render a list of papers with their details."""
    if df.empty:
        st.info(f"No papers found in {category}.")
        return

    # Sort options
    sort_by = st.selectbox(
        "Sort by",
        ["abstract_score", "title", "title_score"],
        format_func=lambda x: "Final Score" if x == "abstract_score" else (
            "Initial Score" if x == "title_score" else "Title"),
        key=f"sort_{category}"
    )

    df_sorted = df.sort_values(by=sort_by, ascending=sort_by not in ["abstract_score", "title_score"])

    for _, paper in df_sorted.iterrows():
        score_display = f"(Initial: {paper.get('title_score', 0):.2f}, Final: {paper.get('abstract_score', 0):.2f})"
        with st.expander(f"{paper['title']} {score_display}"):
            if 'topics' in paper and paper['topics']:
                st.write("**Topics:**", ", ".join(paper['topics']))
            if 'abstract' in paper and paper['abstract']:
                st.write("**Abstract:**", paper['abstract'])

            if paper.get('metadata'):
                if 'key_findings' in paper['metadata']:
                    st.write("**Key Findings:**", paper['metadata']['key_findings'])
                if 'industrial_applications' in paper['metadata']:
                    st.write("**Industrial Applications:**")
                    for app in paper['metadata']['industrial_applications']:
                        st.write(f"- {app}")

            if 'link' in paper and paper['link']:
                st.markdown(f"[View Paper]({paper['link']})")


def create_paper_viewer():
    st.title("Industrial Visual Inspection Paper Analyzer")

    # Sidebar for file selection and filtering
    st.sidebar.title("Controls")

    # Determine storage mode
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPOSITORY")
    use_github = (github_token is not None) and (github_repo is not None)

    # Find analysis result files
    if use_github:
        try:
            owner, repo = github_repo.split('/')
            storage = StorageManager(owner, repo)
            result_files = storage.list_analysis_files()
            if not result_files:
                st.warning("No analysis results found in GitHub. Falling back to local storage.")
                use_github = False
        except Exception as e:
            st.error(f"Error accessing GitHub storage: {e}")
            use_github = False

    if not use_github:
        results_dir = Path("analysis_results")
        if not results_dir.exists():
            st.error("No analysis results found. Please run the analyzer first.")
            return

        result_files = list(results_dir.glob("papers_analysis_*.json"))
        if not result_files:
            st.error("No analysis results found. Please run the analyzer first.")
            return

    # Storage mode indicator
    st.sidebar.info("Using " + ("GitHub storage" if use_github else "local storage"))

    # File selection
    selected_file = st.sidebar.selectbox(
        label="Select Analysis Results",
        options=result_files,
        format_func=lambda x: f"Analysis from {x['key'].split('_')[-1].replace('.json', '')}" if use_github
        else f"Analysis from {x.stem.split('_')[-1]}"
    )

    # Load selected results
    if use_github:
        results = storage.get_analysis_file(selected_file['download_url'])
    else:
        results = load_analysis_results(selected_file)

    # Convert all paper categories to DataFrames with proper ordering
    papers_data = {
        "Top Papers": pd.DataFrame(results.get("top_papers", [])),
        "Relevant Papers": pd.DataFrame(results.get("relevant_papers", [])),
        "Other Papers": pd.DataFrame(results.get("other_papers", []))
    }

    # Global filters in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Filters")

    # Collect all topics across all paper categories
    all_topics = set()
    for df in papers_data.values():
        if not df.empty and 'topics' in df.columns:
            for topics in df['topics']:
                if isinstance(topics, list):
                    all_topics.update(topics)

    # Topic filter
    if all_topics:
        selected_topics = st.sidebar.multiselect(
            "Filter by Topics",
            sorted(all_topics)
        )

        # Apply topic filter to all dataframes
        if selected_topics:
            for category in papers_data:
                df = papers_data[category]
                if not df.empty and 'topics' in df.columns:
                    papers_data[category] = df[df['topics'].apply(
                        lambda x: any(topic in x for topic in selected_topics)
                    )]

    # Create main tabs for different views
    view_tab, analytics_tab, how_it_works_tab = st.tabs(["Papers View", "Analytics", "How it Works"])

    with view_tab:
        # Create tabs for different paper categories
        paper_tabs = st.tabs(list(papers_data.keys()))

        # Render each category in its respective tab
        for tab, category in zip(paper_tabs, papers_data):
            with tab:
                df = papers_data[category]
                st.subheader(f"{category} ({len(df)} papers)")
                render_papers_list(df, category)

    with analytics_tab:
        # Combined analytics across all categories
        st.subheader("Analysis Overview")

        # Paper counts
        counts = {cat: len(df) for cat, df in papers_data.items() if not df.empty}
        if counts:
            fig_counts = px.pie(
                values=list(counts.values()),
                names=list(counts.keys()),
                title="Distribution of Papers by Category"
            )
            st.plotly_chart(fig_counts)
        else:
            st.info("No papers found in any category.")

        # Score distributions
        dfs_with_scores = {
            cat: df for cat, df in papers_data.items()
            if not df.empty and 'abstract_score' in df.columns
        }

        if dfs_with_scores:
            # Combine all papers with their categories
            score_data = []
            for cat, df in dfs_with_scores.items():
                cat_scores = df[['abstract_score']].copy()
                cat_scores['Category'] = cat
                score_data.append(cat_scores)

            combined_scores = pd.concat(score_data)

            fig_scores = px.box(
                combined_scores,
                x='Category',
                y='abstract_score',
                title='Score Distribution by Category'
            )
            st.plotly_chart(fig_scores)

        # Topic distribution across all papers
        if all_topics:
            topic_counts = {}
            for df in papers_data.values():
                if not df.empty and 'topics' in df.columns:
                    for topics in df['topics']:
                        if isinstance(topics, list):
                            for topic in topics:
                                topic_counts[topic] = topic_counts.get(topic, 0) + 1

            topic_df = pd.DataFrame(
                {'Topic': list(topic_counts.keys()), 'Count': list(topic_counts.values())}
            ).sort_values('Count', ascending=False)

            fig_topics = px.bar(
                topic_df,
                x='Topic',
                y='Count',
                title='Overall Topic Distribution'
            )
            st.plotly_chart(fig_topics)


    with how_it_works_tab:
        st.header("How the Analysis Works")

        st.write("""
        This application analyzes research papers from arXiv's Computer Vision feed (http://export.arxiv.org/rss/cs.CV) 
        to identify and evaluate papers relevant to industrial visual inspection. Here's how the process works:
    
        #### 1. Initial Screening
        First, we evaluate each paper's title using GPT-4o-mini with a specialized prompt that focuses on industrial 
        visual inspection criteria, including computer vision techniques, defect detection, quality control automation, 
        and industrial applications. This generates the **Initial Score** you see for each paper.
    
        #### 2. Detailed Analysis
        Papers that pass the initial screening have their abstracts evaluated, resulting in a **Final Score**. 
        The most promising papers (those with high final scores) undergo a detailed analysis where we:
        - Extract key topics and techniques
        - Identify potential industrial applications
        - Summarize key findings
    
        #### 3. Paper Categories
        Papers are sorted into three categories:
        - **Top Papers**: Highly relevant papers that passed both title and abstract screening
        - **Relevant Papers**: Papers that showed promise in title screening but didn't meet the abstract threshold
        - **Other Papers**: Papers that didn't pass the initial title screening
    
        #### Using the Interface
        - Use the **Papers View** tab to explore individual papers, their scores, and detailed analyses
        - The **Analytics** tab provides visualizations of the overall analysis results
        - Use the sidebar filters to focus on specific topics of interest
        - Sort options allow you to organize papers by different criteria within each category
    
        #### Implementation Details
        In case you are interested in technical details or want to customize the project to your needs, have a look at
        the project's GitHub repository (https://github.com/MRaffl98/research-analyzer).
        """)


if __name__ == "__main__":
    create_paper_viewer()