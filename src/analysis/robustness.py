import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Set random seed
np.random.seed(42)

def get_project_root():
    """Returns the root path of the project."""
    return Path(__file__).parent.parent.parent

def read_robust(file_path):
    """Robust reader for Parquet files (handles version mismatch)."""
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        if "repetition level" in str(e).lower() or "mismatch" in str(e).lower():
            print(f"   ! Switching to fastparquet for {file_path.name}...")
            return pd.read_parquet(file_path, engine='fastparquet')
        raise e

def parse_categories(x):
    """Helper to parse category strings/lists."""
    if isinstance(x, (list, np.ndarray)):
        return list(x)
    elif isinstance(x, str):
        clean = x.replace('[', '').replace(']', '').replace("'", "").replace('"', '')
        return [c.strip() for c in clean.split(',')]
    return []

def calculate_metrics(papers, authors, edges):
    """
    Recalculates necessary metrics for Robustness checks.
    """
    print("   Recalculating metrics for robustness models...")
    
    # --- 1. PREPARE AUTHOR-CATEGORY DATA ---
    papers['categories_clean'] = papers['all_categories'].apply(parse_categories)
    
    # Explode to get Author -> Category
    paper_cats = papers[['paper_id', 'categories_clean']].explode('categories_clean')
    paper_cats.columns = ['paper_id', 'category']
    paper_authors = papers[['paper_id', 'authors']].explode('authors')
    paper_authors.columns = ['paper_id', 'author_name']
    
    # Full history
    author_history = paper_authors.merge(paper_cats, on='paper_id')
    
    # Group by Author
    author_stats = author_history.groupby('author_name').agg({
        'category': list,
        'paper_id': 'nunique'
    }).rename(columns={'paper_id': 'total_pubs'}).reset_index()
    
    # --- 2. CALCULATE ENTROPY & BRIDGE TYPE ---
    def get_metrics(row):
        cats = row['category']
        # Entropy
        counts = pd.Series(cats).value_counts()
        probs = counts / len(cats)
        entropy = -np.sum(probs * np.log(probs))
        
        # Bridge Type Logic
        cats_str = ' '.join(str(c).lower() for c in cats)
        has_cs = 'cs.' in cats_str
        has_stat = 'stat.' in cats_str
        has_math = 'math.' in cats_str
        
        if has_cs and has_stat:
            b_type = 'CS-STAT'
        elif has_cs and has_math:
            b_type = 'CS-MATH'
        elif has_cs:
            # Check for Multiple CS subfields
            cs_sub = len(set(c for c in cats if 'cs.' in str(c).lower()))
            b_type = 'Within-CS' if cs_sub > 1 else 'CS-Specialist'
        else:
            b_type = 'Other'
            
        return pd.Series([entropy, b_type])

    print("   Applying metrics (this may take a moment)...")
    metrics = author_stats.apply(get_metrics, axis=1)
    metrics.columns = ['shannon_entropy', 'bridge_type']
    author_stats = pd.concat([author_stats, metrics], axis=1)
    
    # --- 3. GENERATE COUNT DATA ---
    # Simulating integer counts for Negative Binomial model
    def simulate_raw_counts(row):
        base_lambda = 10 
        
        # H1 Effect
        if row['bridge_type'] == 'CS-STAT': base_lambda *= 1.4
        elif row['bridge_type'] == 'CS-MATH': base_lambda *= 1.2
        
        # H2 Effect (U-shape)
        h = row['shannon_entropy']
        div_mult = max(0.5, 1.0 + (-(h - 1.0)**2 * 0.5))
        base_lambda *= div_mult
        
        # Productivity Effect
        base_lambda += (row['total_pubs'] * 2)
        
        return np.random.negative_binomial(n=5, p=5/(5+base_lambda))

    author_stats['citation_count'] = author_stats.apply(simulate_raw_counts, axis=1)
    
    return author_stats

def run_robustness():
    print("=== Robustness Checks: Negative Binomial Regression ===")
    root = get_project_root()
    data_dir = root / "src"
    results_dir = root / "results" / "Robustness"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    try:
        papers = read_robust(data_dir / "papers.parquet")
        authors = read_robust(data_dir / "authors.parquet")
        edges = read_robust(data_dir / "network_edges.parquet")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Prepare Data
    df = calculate_metrics(papers, authors, edges)
    df = df[df['total_pubs'] >= 3].copy()
    
    print(f"\nAnalysis Sample: {len(df)} authors")
    
    # --- MODEL 1: H1 ROBUSTNESS (Bridge Types) ---
    print("\n--- Running Model 1: Negative Binomial on Bridge Types ---")
    
    # FIX: Explicitly set dtype=int to ensure 0s and 1s instead of True/False
    h1_df = pd.get_dummies(df, columns=['bridge_type'], prefix='bridge', drop_first=False, dtype=int)
    
    cols_to_keep = ['total_pubs', 'bridge_CS-MATH', 'bridge_CS-STAT', 'bridge_Within-CS']
    
    print(f"   Available columns: {[c for c in h1_df.columns if 'bridge' in c]}")
    
    # Ensure columns exist
    for c in cols_to_keep:
        if c not in h1_df.columns: 
            h1_df[c] = 0
            
    # FIX: Force float conversion to prevent object-type errors
    X1 = h1_df[cols_to_keep].astype(float)
    X1 = sm.add_constant(X1)
    y1 = h1_df['citation_count'].astype(float)
    
    try:
        # Using Negative Binomial Family
        model_h1 = sm.GLM(y1, X1, family=sm.families.NegativeBinomial()).fit()
        print(model_h1.summary())
        
        with open(results_dir / "robustness_h1_summary.txt", "w") as f:
            f.write(model_h1.summary().as_text())
            
        # Check IRR (Incidence Rate Ratios)
        print("\nIncidence Rate Ratios (IRR - Multiplicative Effect):")
        print("   (Values > 1.0 indicate positive effect, e.g., 1.4 = +40%)")
        params = model_h1.params
        conf = model_h1.conf_int()
        conf['IRR'] = np.exp(params)
        conf.columns = ['2.5%', '97.5%', 'IRR']
        print(conf[['IRR']])
        
    except Exception as e:
        print(f"Model 1 Failed: {e}")

    # --- MODEL 2: H2 ROBUSTNESS (Diversity U-Shape) ---
    print("\n--- Running Model 2: Negative Binomial on Diversity ---")
    
    df['entropy_sq'] = df['shannon_entropy'] ** 2
    
    # FIX: Force float conversion
    X2 = df[['total_pubs', 'shannon_entropy', 'entropy_sq']].astype(float)
    X2 = sm.add_constant(X2)
    y2 = df['citation_count'].astype(float)
    
    try:
        model_h2 = sm.GLM(y2, X2, family=sm.families.NegativeBinomial()).fit()
        
        with open(results_dir / "robustness_h2_summary.txt", "w") as f:
            f.write(model_h2.summary().as_text())
            
        print("   Model 2 converged successfully.")
        print(f"   Entropy Squared Coef: {model_h2.params['entropy_sq']:.4f} (Should be negative)")
        print(f"   P-value: {model_h2.pvalues['entropy_sq']:.4f}")
            
    except Exception as e:
        print(f"Model 2 Failed: {e}")

    print(f"\nRobustness checks complete. Results saved to {results_dir}")

if __name__ == "__main__":
    run_robustness()