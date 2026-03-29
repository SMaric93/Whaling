"""
Compile Authoritative Results

Reads all CSV tables generated during the post-connectivity rebuild
and compiles them into a single Markdown and LaTeX document for easy
copying into the paper draft.
"""

import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
TABLES_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity' / 'tables'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'post_connectivity'

# List of tables to include cleanly
TABLES_TO_COMPILE = [
    ("connectivity_summary.csv", "Phase 0: Canonical Connected Set Summary"),
    ("kss_variance_decomposition_authoritative.csv", "Phase 1: KSS Variance Decomposition (Authoritative)"),
    ("route_choice_hierarchy.csv", "Phase 4: Route Choice Hierarchy (Destination Routing)"),
    ("vessel_mover_power.csv", "Phase 4: Vessel Mover Test (Hardware Setup vs Manager)"),
    ("production_surface_submodularity.csv", "Phase 4: Production Surface & Structural Submodularity"),
    ("floor_raising_insurance.csv", "Phase 4: Floor-Raising & Managerial Insurance (Quantile)"),
    ("matching_welfare_counterfactuals.csv", "Phase 4: Matching Welfare Counterfactuals (Optimal Assignments)"),
    ("offpolicy_diagnostics_smd.csv", "Phase 4: Off-Policy Evaluation Diagnostics (Covariate Balance)"),
    ("mechanism_crew_network.csv", "Phase 4: Mechanisms (Crew Selection & Network Portfolios)"),
    ("stopping_hazard_authoritative.csv", "Phase 3: Stopping Hazard (Patch Exhaustion)"),
]

def format_number(x):
    if isinstance(x, float):
        if abs(x) < 0.001 and x != 0:
            return f"{x:.2e}"
        return f"{x:.4f}"
    return x

def compile_master_tables():
    print(f"Compiling results from {TABLES_DIR}...")
    
    md_content = ["# Authoritative Post-Connectivity Rerun Tables\n"]
    tex_content = [
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage{geometry}",
        "\\geometry{margin=1in}",
        "\\begin{document}\n",
        "\\section*{Authoritative Post-Connectivity Rerun Tables}\n"
    ]
    
    for filename, title in TABLES_TO_COMPILE:
        path = TABLES_DIR / filename
        if not path.exists():
            print(f"Skipping {filename} (File not found)")
            continue
            
        try:
            df = pd.read_csv(path)
            # Format numbers safely
            df_fmt = df.applymap(format_number)
            
            # Markdown output
            md_content.append(f"## {title}\n")
            md_content.append(f"`{filename}`\n")
            md_content.append(df_fmt.to_markdown(index=False))
            md_content.append("\n\n---\n")
            
            # LaTeX output
            tex_content.append(f"\\subsection*{{{title}}}")
            # Ensure safe latex formatting
            tex_table = df_fmt.to_latex(index=False, escape=True, column_format='l' + 'r' * (len(df.columns) - 1))
            tex_content.append("\\begin{table}[h!]")
            tex_content.append("\\centering")
            tex_content.append(tex_table)
            tex_content.append(f"\\caption{{{title}}}")
            tex_content.append("\\end{table}\n")
            tex_content.append("\\vspace{1em}\n")
            
        except Exception as e:
            print(f"Error compiling {filename}: {e}")
            
    tex_content.append("\\end{document}\n")
    
    md_path = OUTPUT_DIR / "master_results.md"
    tex_path = OUTPUT_DIR / "master_results.tex"
    
    with open(md_path, "w") as f:
        f.write("\n".join(md_content))
        
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_content))
        
    print(f"\nSUCCESS: Compiled Markdown tables to {md_path}")
    print(f"SUCCESS: Compiled LaTeX tables to {tex_path}")

if __name__ == "__main__":
    compile_master_tables()
