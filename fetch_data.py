"""
Fetch GitHub issues from PyTorch repository.
"""

import requests
import time
import math
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def fetch_issues(
    owner="pytorch",
    repo="pytorch", 
    num_issues=10_000,
    rate_limit=10_000,
    issues_path=Path("."),
    github_token=None
):
    """
    Fetch GitHub issues from a repository.
    
    Args:
        owner: Repository owner
        repo: Repository name
        num_issues: Number of issues to fetch
        rate_limit: Internal rate limit for batch processing
        issues_path: Directory to save issues
        github_token: GitHub API token (optional)
    
    Returns:
        DataFrame of fetched issues
    """
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    # Set up headers
    headers = {}
    if github_token:
        headers = {"Authorization": f"token {github_token}"}

    batch = []
    all_issues = []
    per_page = 100
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    print(f"Attempting to fetch {num_issues} issues from {owner}/{repo}")
    print(f"Number of pages to request based on num_issues: {num_pages}")
    print(f"Internal rate limit for batch processing set to: {rate_limit} issues")

    for page in tqdm(range(1, num_pages + 1)):
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues_response = requests.get(
            f"{base_url}/{owner}/{repo}/{query}",
            headers=headers
        )

        print(f"--- Page {page} ---")
        print(f"Status Code: {issues_response.status_code}")

        if issues_response.status_code == 200:
            current_page_issues = issues_response.json()
            print(f"Issues received on this page: {len(current_page_issues)}")
            if not current_page_issues:
                print(f"No more issues found from page {page}. Stopping fetch.")
                break
            batch.extend(current_page_issues)
        else:
            print(f"Error fetching issues on page {page}. Response text: {issues_response.text}")
            break

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []
            print(f"Reached internal rate limit ({rate_limit} issues processed). Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)

    # Convert 'labels' and 'body' columns to string representation
    df['labels'] = df['labels'].apply(lambda x: str(x))
    df['body'] = df['body'].apply(lambda x: str(x) if x is not None else "")

    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(f"Downloaded all the issues for {repo}! Total issues collected: {len(df)}. Dataset stored at {issues_path}/{repo}-issues.jsonl")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch PyTorch GitHub issues")
    parser.add_argument("--owner", default="pytorch", help="Repository owner")
    parser.add_argument("--repo", default="pytorch", help="Repository name")
    parser.add_argument("--num-issues", type=int, default=10_000, help="Number of issues to fetch")
    parser.add_argument("--rate-limit", type=int, default=10_000, help="Internal rate limit")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--github-token", default=None, help="GitHub API token")
    
    args = parser.parse_args()
    
    fetch_issues(
        owner=args.owner,
        repo=args.repo,
        num_issues=args.num_issues,
        rate_limit=args.rate_limit,
        issues_path=Path(args.output_dir),
        github_token=args.github_token
    )
