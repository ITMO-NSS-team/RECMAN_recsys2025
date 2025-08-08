import pandas as pd

# Load CSV and replace missing with zeros
df = pd.read_csv('experimental_data.csv')

# Models and metrics to include
models = ['SASRec', 'HypSASRec', 'SASRec_man']
metrics = ['HR@10', 'MRR@10', 'NDCG@10', 'COV@10']

# Begin constructing LaTeX lines
lines = [
    r"\begin{table}[ht!]",
    r"\centering",
    r"\caption{Evaluation results and relative improvements for SASRec architecture}",
    r"\begin{tabular}{cccccc}",
    r" &  & HR@10 & MRR@10 & NDCG@10 & COV@10  \\",
    r"\midrule"
]

# Group by dataset and build rows
for dataset, group in df.groupby('dataset'):
    # select and fill missing models
    sub = group.set_index('model').reindex(models).fillna(0)
    # determine best value per metric
    best_vals = {m: sub[m].max() for m in metrics}
    
    lines.append(r"\multirow{4}{*}{\rotatebox[origin=c]{90}{" + dataset + "}}  ")
    # output each model row, bold per-metric if equals best_vals
    for model in models:
        row = sub.loc[model]
        cells = []
        for metric in metrics:
            val = row[metric]
            formatted = f"{val:.3f}"
            if val == best_vals[metric]:
                formatted = r"\textbf{" + formatted + "}"
            cells.append(formatted)
        lines.append(f"& {model} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")
    
    # relative improvements of SASRec vs best per metric
    base = sub.loc['SASRec']
    rels = []
    for metric in metrics:
        best = best_vals[metric]
        base_val = base[metric] if base[metric] != 0 else 1e-9
        rel = (best - base_val) / base_val * 100
        rels.append(f"+{rel:.0f}\\%")
    lines.append("& SASRec vs best  & " + " & ".join(rels) + r" \\")
    lines.append(r"\midrule")

# End table
lines += [r"\end{tabular}",r"\label{tab:sasrec_results}", r"\end{table}"]

# Write to .tex
with open('../../hypsasrec/Hyperbolic-SASRec/table1.tex', 'w') as f:
    f.write("\n".join(lines))

print("LaTeX table saved to table1.tex")
