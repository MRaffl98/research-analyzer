import requests
import json
from datetime import datetime
from typing import List, Dict
import os


class StorageManager:
    def __init__(self, repo_owner: str, repo_name: str):
        """Initialize storage manager with GitHub repository details."""
        self.owner = repo_owner
        self.repo = repo_name
        self.token = os.getenv('GITHUB_TOKEN')
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.api_base = f'https://api.github.com/repos/{repo_owner}/{repo_name}'

    def save_analysis_results(self, top_papers: List, relevant_papers: List,
                              other_papers: List) -> str:
        """Save analysis results as a GitHub release asset."""
        timestamp = datetime.now().strftime("%Y-%m-%d")

        # Prepare the results
        results = {
            "timestamp": timestamp,
            "top_papers": [paper.__dict__ for paper in top_papers],
            "relevant_papers": [paper.__dict__ for paper in relevant_papers],
            "other_papers": [paper.__dict__ for paper in other_papers]
        }

        # Create the release
        release_data = {
            'tag_name': f'analysis-{timestamp}',
            'name': f'Analysis Results {timestamp}',
            'body': f'Paper analysis results for {timestamp}',
            'draft': False,
            'prerelease': False
        }

        response = requests.post(
            f'{self.api_base}/releases',
            headers=self.headers,
            json=release_data
        )
        release = response.json()

        # Upload the asset
        upload_url = release['upload_url'].replace('{?name,label}', '')
        files = {
            'file': (
                f'papers_analysis_{timestamp}.json',
                json.dumps(results, indent=2),
                'application/json'
            )
        }
        response = requests.post(
            f"{upload_url}?name=papers_analysis_{timestamp}.json",
            headers={**self.headers, 'Content-Type': 'application/json'},
            data=json.dumps(results)
        )

        return response.json()['browser_download_url']

    def list_analysis_files(self) -> List[Dict]:
        """List all analysis files from GitHub releases."""
        response = requests.get(
            f'{self.api_base}/releases',
            headers=self.headers
        )
        releases = response.json()

        files = []
        for release in releases:
            for asset in release['assets']:
                if asset['name'].endswith('.json'):
                    files.append({
                        'key': asset['name'],
                        'date': asset['created_at'],
                        'size': asset['size'],
                        'download_url': asset['browser_download_url']
                    })

        return sorted(files, key=lambda x: x['date'], reverse=True)

    def get_analysis_file(self, download_url: str) -> Dict:
        """Get a specific analysis file from GitHub release."""
        response = requests.get(download_url)
        if response.status_code == 200:
            return response.json()
        return {}
