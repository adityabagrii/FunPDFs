# import os
# import csv
# import arxiv
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed

# client = arxiv.Client()

# def download_paper(r, idx, dirpath, query):
#     filename = f"{idx}.pdf"
#     filepath = os.path.join(dirpath, filename)
    
#     if not os.path.exists(filepath):
#         try:
#             r.download_pdf(dirpath, filename=filename)
#         except Exception as e:
#             print(f"Failed to download paper '{r.title}': {e}")
#             return None
    
#     return {
#         'filename': filename,
#         'title': r.title,
#         'authors': ', '.join(author.name for author in r.authors),
#         'published': r.published.strftime('%Y-%m-%d'),
#         'url': r.entry_id,
#         'query': query
#     }

# def download_search(query, dirpath, max_results=50, max_workers=5):
#     os.makedirs(dirpath, exist_ok=True)
#     metadata = []

#     search = arxiv.Search(
#         query=query,
#         max_results=max_results,
#         sort_by=arxiv.SortCriterion.SubmittedDate
#     )

#     results = list(client.results(search))

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(download_paper, r, idx, dirpath, query): idx
#             for idx, r in enumerate(results, start=1)
#         }

#         for future in tqdm(as_completed(futures), total=len(futures),
#                            desc=f"Downloading papers for query: '{query}'"):
#             result = future.result()
#             if result:
#                 metadata.append(result)

#     # Save metadata to query-specific CSV
#     csv_path = os.path.join(dirpath, 'metadata.csv')
#     with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.DictWriter(f, fieldnames=['filename', 'title', 'authors', 'published', 'url', 'query'])
#         writer.writeheader()
#         writer.writerows(metadata)

#     print(f"Metadata saved at {csv_path}")
#     return metadata


# def main():
#     queries = [
#         # Computer Science
#         "deep learning for computer vision",
#         "graph neural networks",
#         "natural language processing",
#         "reinforcement learning",
#         "privacy preserving machine learning",

#         # Physics
#         "quantum computing algorithms",
#         "dark matter and dark energy",
#         "gravitational waves detection",
#         "high energy particle physics",

#         # Biology & Medicine
#         "genomics and bioinformatics",
#         "CRISPR gene editing",
#         "neuroscience brain mapping",
#         "cancer immunotherapy",

#         # Environmental Science
#         "climate change impact on biodiversity",
#         "sustainable energy solutions",
#         "ocean acidification effects",

#         # Economics & Finance
#         "financial fraud detection",
#         "cryptocurrency market dynamics",
#         "behavioral economics in decision making",

#         # Social Science & Psychology
#         "social media and mental health",
#         "cognitive behavioral therapy effectiveness",
#         "human migration patterns and policies",

#         # Chemistry & Material Science
#         "nanomaterials for energy storage",
#         "organic solar cells development",
#         "green chemistry innovations",

#         # Engineering & Robotics
#         "robotics path planning",
#         "autonomous vehicles safety",
#         "3D printing in biomedical engineering"
#     ]

#     base_dir = "dataset"
#     os.makedirs(base_dir, exist_ok=True)

#     all_metadata = []

#     for query in queries:
#         safe_query = query.replace(" ", "_").replace("/", "-")
#         dirpath = os.path.join(base_dir, safe_query)
#         metadata = download_search(query, dirpath, max_results=50, max_workers=5)
#         all_metadata.extend(metadata)

#     # Save global metadata
#     global_csv_path = os.path.join(base_dir, 'global_metadata.csv')
#     with open(global_csv_path, mode='w', newline='', encoding='utf-8') as f:
#         writer = csv.DictWriter(f, fieldnames=['filename', 'title', 'authors', 'published', 'url', 'query'])
#         writer.writeheader()
#         writer.writerows(all_metadata)

#     print(f"\nGlobal metadata saved at {global_csv_path}")


# if __name__ == '__main__':
#     main()

import os
import re
import tarfile
import arxiv
import json
import shutil
from tqdm import tqdm

client = arxiv.Client()

def download_and_parse_source(r, idx, dirpath, query):
    filename = f"{idx}_source.tar.gz"
    filepath = os.path.join(dirpath, filename)

    try:
        r.download_source(dirpath=dirpath, filename=filename)
    except Exception as e:
        print(f"Failed to download LaTeX source for '{r.title}': {e}")
        return None

    # Extract tar.gz
    extracted_dir = os.path.join(dirpath, f"{idx}_source")
    os.makedirs(extracted_dir, exist_ok=True)

    try:
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=extracted_dir)
    except Exception as e:
        print(f"Failed to extract {filename}: {e}")
        return None
    finally:
        # Delete the .tar.gz file after extraction
        if os.path.exists(filepath):
            os.remove(filepath)

    # Parse all .tex files
    outline = []
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            if file.endswith('.tex'):
                tex_path = os.path.join(root, file)
                try:
                    with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        sections = parse_latex_headings(content)
                        outline.extend(sections)
                except Exception as e:
                    print(f"Error reading {tex_path}: {e}")

    # Delete the extracted source directory after parsing
    shutil.rmtree(extracted_dir, ignore_errors=True)

    if not outline:
        return None

    return {
        'title': r.title,
        'outline': outline,
        'query': query,
        'arxiv_id': r.get_short_id(),
        'url': r.entry_id
    }


def parse_latex_headings(content):
    pattern = re.compile(r'(\\section\*?\{(.*?)\}|\\subsection\*?\{(.*?)\}|\\subsubsection\*?\{(.*?)\})', re.DOTALL)
    matches = pattern.findall(content)

    outline = []
    for match in matches:
        if match[0].startswith(r'\section'):
            level = 'H1'
            text = match[1]
        elif match[0].startswith(r'\subsection'):
            level = 'H2'
            text = match[2]
        elif match[0].startswith(r'\subsubsection'):
            level = 'H3'
            text = match[3]
        else:
            continue

        outline.append({
            'level': level,
            'text': text.strip()
        })

    return outline


def download_search(query, dirpath, max_results=50):
    os.makedirs(dirpath, exist_ok=True)
    metadata = []

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results = list(client.results(search))

    for idx, r in enumerate(tqdm(results, desc=f"Processing query: {query}"), start=1):
        paper_data = download_and_parse_source(r, idx, dirpath, query)
        if paper_data:
            metadata.append(paper_data)

    json_path = os.path.join(dirpath, 'metadata.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata for query '{query}' saved at {json_path}")
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

    base_dir = "dataset_json"
    os.makedirs(base_dir, exist_ok=True)

    all_metadata = []

    for query in queries:
        safe_query = query.replace(" ", "_").replace("/", "-")
        dirpath = os.path.join(base_dir, safe_query)
        metadata = download_search(query, dirpath, max_results=50)
        all_metadata.extend(metadata)

    global_json_path = os.path.join(base_dir, 'global_metadata.json')
    with open(global_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"\nGlobal metadata saved at {global_json_path}")


if __name__ == '__main__':
    main()