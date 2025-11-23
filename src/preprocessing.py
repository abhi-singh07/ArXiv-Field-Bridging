import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re


def load_arxiv_metadata(filepath, start_year=2010, end_year=2023):
    """
    Load ArXiv metadata from JSONL file.
    
    Args:
        filepath: Path to arxiv-metadata-oai-snapshot.json
        start_year: Start year filter
        end_year: End year filter
    
    Returns:
        pd.DataFrame with filtered papers
    """
    print(f"Loading ArXiv metadata from {filepath}...")
    data = []
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            try:
                paper = json.loads(line)
                
                # Extract year from versions
                if paper.get('versions') and len(paper['versions']) > 0:
                    created = paper['versions'][0].get('created', '')
                    parts = created.split()
                    if len(parts) >= 4:
                        year = int(parts[3])
                        
                        if start_year <= year <= end_year:
                            data.append(paper)
                
                # Progress
                if (i + 1) % 100000 == 0:
                    print(f"  Processed {i+1:,} lines, found {len(data):,} papers")
                    
            except Exception as e:
                continue
    
    print(f"Total papers loaded: {len(data):,}\n")
    return pd.DataFrame(data)


def parse_authors(authors_parsed):
    """
    Extract clean author names from authors_parsed field.
    
    Args:
        authors_parsed: List of [last, first, middle] lists
    
    Returns:
        List of "FirstName LastName" strings
    """
    if not authors_parsed or not isinstance(authors_parsed, list):
        return []
    
    clean_authors = []
    for author in authors_parsed:
        if not isinstance(author, list) or len(author) < 2:
            continue
        
        # author format: [last, first, middle]
        last = author[0].strip()
        first = author[1].strip()
        
        if last and first:
            # Create "First Last" format
            full_name = f"{first} {last}"
            clean_authors.append(full_name)
    
    return clean_authors


def extract_primary_category(categories):
    """Extract first category as primary."""
    if not categories:
        return None
    cats = categories.split()
    return cats[0] if cats else None


def extract_all_categories(categories):
    """Extract all categories as list."""
    if not categories:
        return []
    return categories.split()


def extract_year(versions):
    """Extract publication year from versions."""
    if not versions or not isinstance(versions, list) or len(versions) == 0:
        return None
    
    try:
        created = versions[0].get('created', '')
        parts = created.split()
        if len(parts) >= 4:
            return int(parts[3])
    except:
        pass
    return None


def process_papers(df, min_authors=1, max_authors=50):
    """
    Process raw paper data into clean papers.parquet.
    
    Args:
        df: Raw dataframe from load_arxiv_metadata
        min_authors: Minimum number of authors
        max_authors: Maximum number of authors (filter noise)
    
    Returns:
        pd.DataFrame with processed paper data
    """
    print("Processing papers...")
    
    papers = []
    
    for idx, row in df.iterrows():
        # Parse authors
        author_list = parse_authors(row.get('authors_parsed'))
        
        # Filter by author count
        if len(author_list) < min_authors or len(author_list) > max_authors:
            continue
        
        # Extract fields
        paper_id = row.get('id')
        title = row.get('title', '').strip()
        abstract = row.get('abstract', '').strip()
        
        # Categories
        primary_cat = extract_primary_category(row.get('categories'))
        all_cats = extract_all_categories(row.get('categories'))
        
        # Year
        year = extract_year(row.get('versions'))
        
        # Skip if missing critical fields
        if not paper_id or not primary_cat or not year or len(author_list) == 0:
            continue
        
        papers.append({
            'paper_id': paper_id,
            'title': title,
            'abstract': abstract,
            'year': year,
            'primary_category': primary_cat,
            'all_categories': all_cats,
            'num_authors': len(author_list),
            'authors': author_list  # List of author names
        })
        
        if (idx + 1) % 50000 == 0:
            print(f"  Processed {idx+1:,} papers, kept {len(papers):,}")
    
    papers_df = pd.DataFrame(papers)
    print(f"Total papers retained: {len(papers_df):,}\n")
    
    return papers_df


def create_author_lookup(papers_df):
    """
    Create author-level data with paper counts and categories.
    
    Args:
        papers_df: Processed papers dataframe
    
    Returns:
        pd.DataFrame with author-level statistics
    """
    print("Creating author lookup...")
    
    author_data = defaultdict(lambda: {
        'papers': [],
        'categories': set(),
        'years': []
    })
    
    # Aggregate author information
    for idx, row in papers_df.iterrows():
        for author in row['authors']:
            author_data[author]['papers'].append(row['paper_id'])
            author_data[author]['categories'].update(row['all_categories'])
            author_data[author]['years'].append(row['year'])
    
    # Convert to dataframe
    authors = []
    for author_name, data in author_data.items():
        authors.append({
            'author_name': author_name,
            'num_papers': len(data['papers']),
            'num_categories': len(data['categories']),
            'categories': sorted(list(data['categories'])),
            'first_year': min(data['years']),
            'last_year': max(data['years']),
            'career_length': max(data['years']) - min(data['years']) + 1
        })
    
    authors_df = pd.DataFrame(authors)
    print(f"Total unique authors: {len(authors_df):,}\n")
    
    return authors_df


def create_network_edges(papers_df):
    """
    Create co-authorship network edges.
    
    Args:
        papers_df: Processed papers dataframe
    
    Returns:
        pd.DataFrame with edge list (author1, author2, weight, papers)
    """
    print("Creating network edges...")
    
    # Track co-authorship relationships
    edges = defaultdict(lambda: {
        'weight': 0,
        'papers': []
    })
    
    # For each paper, create edges between all author pairs
    for idx, row in papers_df.iterrows():
        authors = row['authors']
        paper_id = row['paper_id']
        
        # Create edges for all pairs
        for i in range(len(authors)):
            for j in range(i+1, len(authors)):
                # Alphabetically order to avoid duplicates
                author1, author2 = sorted([authors[i], authors[j]])
                
                edge_key = (author1, author2)
                edges[edge_key]['weight'] += 1
                edges[edge_key]['papers'].append(paper_id)
        
        if (idx + 1) % 50000 == 0:
            print(f"  Processed {idx+1:,} papers, {len(edges):,} edges")
    
    # Convert to dataframe
    edge_list = []
    for (author1, author2), data in edges.items():
        edge_list.append({
            'author1': author1,
            'author2': author2,
            'weight': data['weight'],
            'num_papers': len(data['papers'])
        })
    
    edges_df = pd.DataFrame(edge_list)
    print(f"Total edges: {len(edges_df):,}\n")
    
    return edges_df


def filter_authors_by_productivity(authors_df, papers_df, edges_df, 
                                   min_papers=3, min_collaborators=1):
    """
    Filter to productive authors and update all dataframes.
    
    Args:
        authors_df: Authors dataframe
        papers_df: Papers dataframe
        edges_df: Edges dataframe
        min_papers: Minimum papers per author
        min_collaborators: Minimum co-authors
    
    Returns:
        Tuple of (filtered_authors, filtered_papers, filtered_edges)
    """
    print(f"Filtering authors (min_papers={min_papers}, min_collaborators={min_collaborators})...")
    
    # Filter authors by paper count
    active_authors = set(authors_df[authors_df['num_papers'] >= min_papers]['author_name'])
    print(f"  Authors with >={min_papers} papers: {len(active_authors):,}")
    
    # Filter by collaborators
    author_collaborators = defaultdict(set)
    for _, row in edges_df.iterrows():
        author_collaborators[row['author1']].add(row['author2'])
        author_collaborators[row['author2']].add(row['author1'])
    
    collaborative_authors = {
        author for author, collabs in author_collaborators.items() 
        if len(collabs) >= min_collaborators
    }
    print(f"  Authors with >={min_collaborators} collaborators: {len(collaborative_authors):,}")
    
    # Intersect filters
    keep_authors = active_authors & collaborative_authors
    print(f"  Authors meeting both criteria: {len(keep_authors):,}")
    
    # Filter dataframes
    filtered_authors = authors_df[authors_df['author_name'].isin(keep_authors)].copy()
    
    # Filter papers to only those with at least one kept author
    filtered_papers = papers_df[
        papers_df['authors'].apply(lambda x: any(a in keep_authors for a in x))
    ].copy()
    
    # Filter edges to only kept authors
    filtered_edges = edges_df[
        (edges_df['author1'].isin(keep_authors)) & 
        (edges_df['author2'].isin(keep_authors))
    ].copy()
    
    print(f"  Final authors: {len(filtered_authors):,}")
    print(f"  Final papers: {len(filtered_papers):,}")
    print(f"  Final edges: {len(filtered_edges):,}\n")
    
    return filtered_authors, filtered_papers, filtered_edges


def save_processed_data(authors_df, papers_df, edges_df, output_dir='data/processed'):
    """
    Save processed dataframes as parquet files.
    
    Args:
        authors_df: Authors dataframe
        papers_df: Papers dataframe
        edges_df: Edges dataframe
        output_dir: Output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Saving processed data to {output_dir}...")
    
    authors_df.to_parquet(f"{output_dir}/authors.parquet", index=False)
    print(f"  Saved authors.parquet ({len(authors_df):,} rows)")
    
    papers_df.to_parquet(f"{output_dir}/papers.parquet", index=False)
    print(f"  Saved papers.parquet ({len(papers_df):,} rows)")
    
    edges_df.to_parquet(f"{output_dir}/network_edges.parquet", index=False)
    print(f"  Saved network_edges.parquet ({len(edges_df):,} rows)")
    
    print("\nProcessing complete!")


def main(input_file='data/raw/arxiv-metadata-oai-snapshot.json',
         output_dir='data/processed',
         start_year=2010,
         end_year=2023,
         min_papers=3,
         min_collaborators=1):
    """
    Main processing pipeline.
    
    Args:
        input_file: Path to raw ArXiv metadata
        output_dir: Directory for processed files
        start_year: Filter start year
        end_year: Filter end year
        min_papers: Minimum papers per author
        min_collaborators: Minimum collaborators per author
    """
    # Load raw data
    df_raw = load_arxiv_metadata(input_file, start_year, end_year)
    
    # Process papers
    papers_df = process_papers(df_raw)
    
    # Create author lookup
    authors_df = create_author_lookup(papers_df)
    
    # Create network edges
    edges_df = create_network_edges(papers_df)
    
    # Filter by productivity
    authors_df, papers_df, edges_df = filter_authors_by_productivity(
        authors_df, papers_df, edges_df, 
        min_papers=min_papers,
        min_collaborators=min_collaborators
    )
    
    # Save
    save_processed_data(authors_df, papers_df, edges_df, output_dir)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Papers: {len(papers_df):,}")
    print(f"Authors: {len(authors_df):,}")
    print(f"Edges: {len(edges_df):,}")
    print(f"Year range: {papers_df['year'].min()} - {papers_df['year'].max()}")
    print(f"Top categories: {papers_df['primary_category'].value_counts().head(5).to_dict()}")
    print(f"Average authors per paper: {papers_df['num_authors'].mean():.2f}")
    print(f"Average papers per author: {authors_df['num_papers'].mean():.2f}")


if __name__ == "__main__":
    # Can configure parameters here or in config.py
    main(
        input_file='data/raw/arxiv-metadata-oai-snapshot.json',
        output_dir='data/processed',
        start_year=2010,
        end_year=2023,
        min_papers=3,
        min_collaborators=1
    )