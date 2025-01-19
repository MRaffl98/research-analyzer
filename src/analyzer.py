import feedparser
import json
import logging
import os
import requests
import time

from bs4 import BeautifulSoup
from dataclasses import dataclass, field, asdict
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from typing import Set, List, Tuple, Dict

from storage_utils import StorageManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResearchPaper:
    title: str
    link: str
    abstract: str = ""
    full_text: str = ""
    title_score: float = 0.0
    abstract_score: float = 0.0
    topics: List[str] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.title, self.link))

    def __eq__(self, other):
        if not isinstance(other, ResearchPaper):
            return False
        return self.title == other.title or self.link == other.link


class ResearchAnalyzer:
    def __init__(self, openai_api_key: str, initial_prompt: str, detailed_prompt: str):
        """
        Initialize the research analyzer.

        Args:
            openai_api_key (str): API key for OpenAI
            initial_prompt (str): System prompt for initial paper screening
            detailed_prompt (str): System prompt for detailed paper analysis
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.gpt_model = "gpt-4o-mini"  # Faster model for initial screening
        self.gpt_model_detailed = "gpt-4o-mini"  # More capable model for detailed analysis
        self.processed_papers: Set[ResearchPaper] = set()
        self.initial_prompt = initial_prompt
        self.detailed_prompt = detailed_prompt

    def api_call_with_retry(self, max_retries: int = 3, initial_delay: int = 2, func=None):
        """Make an API call with retry logic."""
        attempt = 0
        while attempt < max_retries:
            try:
                result = func()
                return result if result is not None else 0.0
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    return 0.0
                delay = initial_delay * (2 ** attempt)
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            attempt += 1
        return 0.0

    def evaluate_paper(self, paper: ResearchPaper, content_type: str = "title") -> float:
        """
        Evaluate a paper's relevance based on title or abstract.

        Args:
            paper (ResearchPaper): Paper to evaluate
            content_type (str): Either "title" or "abstract"

        Returns:
            float: Relevance score between 0 and 1
        """

        def _make_call():
            try:
                content = paper.title if content_type == "title" else paper.abstract
                response = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=[
                        {"role": "system", "content": self.initial_prompt},
                        {"role": "user", "content": f"Based on the {content_type}, rate relevance:\n{content}"}
                    ],
                    max_tokens=4,
                    temperature=0.0
                )
                score = float(response.choices[0].message.content)
                return score if 0 <= score <= 1 else 0.0
            except Exception as e:
                logger.warning(f"Could not extract relevance score: {str(e)}")
                return 0.0

        return self.api_call_with_retry(func=_make_call)

    def analyze_paper_content(self, paper: ResearchPaper) -> Dict:
        """
        Perform detailed analysis of paper content based on the detailed prompt.

        Args:
            paper (ResearchPaper): Paper with full text to analyze

        Returns:
            Dict: Analysis results as specified in the detailed prompt
        """

        def _make_call():
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_model_detailed,
                    messages=[
                        {"role": "system", "content": self.detailed_prompt},
                        {"role": "user", "content": f"Analyze this research paper:\n\nTitle: {paper.title}\n\nAbstract: {paper.abstract}\n\nContent: {paper.full_text[:10000]}"}
                    ],
                    max_tokens=1000,
                    temperature=0.0
                )
                import json
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                logger.warning(f"Could not analyze paper content: {str(e)}")
                return {}

        return self.api_call_with_retry(func=_make_call)

    def get_paper_content(self, paper_link: str) -> str:
        """Extract full text content from an arXiv paper."""
        try:
            arxiv_id = paper_link.split('/')[-1]
            html_url = f"https://arxiv.org/html/{arxiv_id}"

            logger.info(f"Fetching HTML content from {html_url}")
            response = requests.get(html_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            main_content = soup.find('main', {'class': 'ltx_page_main'}) or soup.find('div', {'class': 'ltx_page_main'})

            if not main_content:
                return ""

            # Extract text while preserving structure
            content_parts = []

            if title := main_content.find('h1'):
                content_parts.append(title.get_text().strip())

            if abstract := main_content.find('div', string='Abstract'):
                if abstract_text := abstract.find_next('div'):
                    content_parts.extend(["\nAbstract:", abstract_text.get_text().strip()])

            for section in main_content.find_all(['h2', 'h3', 'p']):
                if text := section.get_text().strip():
                    content_parts.append(text)

            return '\n\n'.join(content_parts)

        except Exception as e:
            logger.error(f"Error fetching paper content: {str(e)}")
            return ""

    def process_papers(self, feed_url: str, title_threshold: float = 0.6, abstract_threshold: float = 0.7) -> Tuple[
        List[ResearchPaper], List[ResearchPaper], List[ResearchPaper]]:
        """
        Process papers from an arXiv RSS feed.

        Args:
            feed_url (str): URL of the arXiv RSS feed
            title_threshold (float): Minimum score for title evaluation
            abstract_threshold (float): Minimum score for abstract evaluation

        Returns:
            Tuple[List[ResearchPaper], List[ResearchPaper]]: (top_papers, relevant_papers, other_papers)
        """
        feed = feedparser.parse(feed_url)
        top_papers = []
        relevant_papers = []
        other_papers = []

        logger.info(f"Processing {len(feed.entries)} papers from {feed_url}")

        for entry in feed.entries[:1000]:
            paper = ResearchPaper(
                title=entry.title,
                link=entry.link,
                abstract=entry.get('summary', '')
            )

            if paper in self.processed_papers:
                continue

            self.processed_papers.add(paper)

            # Initial screening using title
            paper.title_score = self.evaluate_paper(paper, "title")
            if paper.title_score < title_threshold:
                other_papers.append(paper)
                continue

            # Detailed screening using abstract
            paper.abstract_score = self.evaluate_paper(paper, "abstract")
            if paper.abstract_score < abstract_threshold:
                relevant_papers.append(paper)
                continue

            # Get full text for highly relevant papers
            if content := self.get_paper_content(paper.link):
                paper.full_text = content

                # Perform detailed analysis
                analysis_results = self.analyze_paper_content(paper)
                paper.topics = analysis_results.get('topics', [])
                paper.metadata = analysis_results

                top_papers.append(paper)
                logger.info(f"Found relevant paper: {paper.title} (Score: {paper.abstract_score:.2f})")

        return top_papers, relevant_papers, other_papers


def create_visual_inspection_analyzer(api_key: str) -> ResearchAnalyzer:
    """Create an analyzer configured for visual inspection research."""
    initial_prompt = """
    You are an expert in industrial visual inspection and quality control systems.
    Rate papers based on their relevance to industrial visual inspection, considering:
    - Computer vision and image processing techniques
    - Defect detection and classification
    - Real-time inspection systems
    - Quality control automation
    - Industrial applications and case studies
    - Application of state of the art machine learning and AI methods for (industrial) computer vision use cases
    
    Return only a float between 0 and 1 so that I can apply the `float()` function on your output."""

    detailed_prompt = """Analyze this research paper's relevance to industrial visual inspection.
    Return a valid raw JSON object (without any Markdown formatting or code blocks) with these fields:
    {
        "topics": ["list of relevant techniques/applications"],
        "relevance_score": float,  # 0-1 scale
        "key_findings": "brief summary of main results",
        "industrial_applications": ["list of potential applications"]
    }"""

    return ResearchAnalyzer(api_key, initial_prompt, detailed_prompt)


def save_analysis_results(top_papers: List, relevant_papers: List, other_papers: List, output_dir: str = "analysis_results"):
    """Save analysis results to a JSON file with timestamp."""
    Path(output_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_file = Path(output_dir) / f"papers_analysis_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "top_papers": [asdict(paper) for paper in top_papers],
        "relevant_papers": [asdict(paper) for paper in relevant_papers],
        "other_papers": [asdict(paper) for paper in other_papers]
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    return output_file


def main():
    # Determine storage mode
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPOSITORY")
    use_github = (github_token is not None) and (github_repo is not None)

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    feed_url = os.getenv("ARXIV_FEED_URL", "http://export.arxiv.org/rss/cs.CV")

    # Create an analyzer for visual inspection (as an example)
    analyzer = create_visual_inspection_analyzer(api_key)

    # Process papers
    top_papers, relevant_papers, other_papers = analyzer.process_papers(feed_url)

    # Save results based on mode
    if use_github:
        try:
            owner, repo = github_repo.split('/')
            storage = StorageManager(owner, repo)
            result_url = storage.save_analysis_results(top_papers, relevant_papers, other_papers)
            print(f"Analysis results saved to GitHub Release: {result_url}")
        except Exception as e:
            print(f"Error saving to GitHub: {e}")
            # Fallback to local storage
            output_file = save_analysis_results(top_papers, relevant_papers, other_papers)
            print(f"Results saved locally as fallback: {output_file}")
    else:
        # Local storage
        output_file = save_analysis_results(top_papers, relevant_papers, other_papers)
        print(f"Analysis results saved locally to: {output_file}")


if __name__ == "__main__":
    main()
