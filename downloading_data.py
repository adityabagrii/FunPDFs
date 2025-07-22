import arxiv
import os
from tqdm import tqdm

def download_papers(category, max_results=100, save_dir='downloads'):
    os.makedirs(save_dir, exist_ok=True)
    
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results = list(search.results())
    for result in tqdm(results, desc=f"Downloading {category}", unit="paper"):
        filename = os.path.join(save_dir, f"{result.get_short_id()}.pdf")
        try:
            result.download_pdf(filename)
        except Exception as e:
            print(f"Failed to download {result.title}: {e}")

# Example categories (you can customize this list)
categories = {
    'physics': 'physics',
    'cs': 'cs',
    'math': 'math',
    'q-bio': 'q-bio',
    'stat': 'stat',
    'eess': 'eess',
    'econ': 'econ'
}

for domain, cat_code in categories.items():
    domain_dir = os.path.join('dataset', domain)
    download_papers(cat_code, max_results=100, save_dir=domain_dir)
