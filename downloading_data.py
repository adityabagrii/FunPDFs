import os
import csv
import arxiv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

client = arxiv.Client()

def download_paper(r, idx, dirpath, query):
    filename = f"{idx}.pdf"
    filepath = os.path.join(dirpath, filename)
    
    if not os.path.exists(filepath):
        try:
            r.download_pdf(dirpath, filename=filename)
        except Exception as e:
            print(f"Failed to download paper '{r.title}': {e}")
            return None
    
    return {
        'filename': filename,
        'title': r.title,
        'authors': ', '.join(author.name for author in r.authors),
        'published': r.published.strftime('%Y-%m-%d'),
        'url': r.entry_id,
        'query': query
    }

def download_search(query, dirpath, max_results=50, max_workers=5):
    os.makedirs(dirpath, exist_ok=True)
    metadata = []

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results = list(client.results(search))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_paper, r, idx, dirpath, query): idx
            for idx, r in enumerate(results, start=1)
        }

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Downloading papers for query: '{query}'"):
            result = future.result()
            if result:
                metadata.append(result)

    # Save metadata to query-specific CSV
    csv_path = os.path.join(dirpath, 'metadata.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'title', 'authors', 'published', 'url', 'query'])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Metadata saved at {csv_path}")
    return metadata


def main():
    queries = [
        # Computer Science
        "deep learning for computer vision",
        "graph neural networks",
        "natural language processing",
        "reinforcement learning",
        "privacy preserving machine learning",

        # Physics
        "quantum computing algorithms",
        "dark matter and dark energy",
        "gravitational waves detection",
        "high energy particle physics",

        # Biology & Medicine
        "genomics and bioinformatics",
        "CRISPR gene editing",
        "neuroscience brain mapping",
        "cancer immunotherapy",

        # Environmental Science
        "climate change impact on biodiversity",
        "sustainable energy solutions",
        "ocean acidification effects",

        # Economics & Finance
        "financial fraud detection",
        "cryptocurrency market dynamics",
        "behavioral economics in decision making",

        # Social Science & Psychology
        "social media and mental health",
        "cognitive behavioral therapy effectiveness",
        "human migration patterns and policies",

        # Chemistry & Material Science
        "nanomaterials for energy storage",
        "organic solar cells development",
        "green chemistry innovations",

        # Engineering & Robotics
        "robotics path planning",
        "autonomous vehicles safety",
        "3D printing in biomedical engineering"
    ]

    base_dir = "dataset"
    os.makedirs(base_dir, exist_ok=True)

    all_metadata = []

    for query in queries:
        safe_query = query.replace(" ", "_").replace("/", "-")
        dirpath = os.path.join(base_dir, safe_query)
        metadata = download_search(query, dirpath, max_results=50, max_workers=5)
        all_metadata.extend(metadata)

    # Save global metadata
    global_csv_path = os.path.join(base_dir, 'global_metadata.csv')
    with open(global_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'title', 'authors', 'published', 'url', 'query'])
        writer.writeheader()
        writer.writerows(all_metadata)

    print(f"\nGlobal metadata saved at {global_csv_path}")


if __name__ == '__main__':
    main()