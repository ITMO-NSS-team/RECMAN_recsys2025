import pandas as pd

# Load CSV and replace missing with zeros
df = pd.read_csv('experimental_data.csv').fillna(0)

# Models and metrics to include for Table 2
models = ['SASRec+', 'HypSASRec+', 'SASRec+_man']
metrics = ['HR@10', 'MRR@10', 'NDCG@10', 'COV@10']

# Begin constructing LaTeX lines for Table 2
lines = [
    r"\begin{table}[ht!]",
    r"\centering",
    r"\caption{Evaluation results and relative improvements for SASRec+ architecture}",
    r"\begin{tabular}{lcccccc}",
    r" &  & HR@10 & MRR@10 & NDCG@10 & COV@10  \\",
    r"\midrule"
]

# Group by dataset and build rows
for dataset, group in df.groupby('dataset'):
    sub = group.set_index('model').reindex(models).fillna(0)
    # determine best value per metric
    best_vals = {m: sub[m].max() for m in metrics}
    
    lines.append(r"\multirow{5}{*}{\rotatebox[origin=c]{90}{" + dataset + "}}  ")
    # output each model row, bold per-metric if equals best_vals
    for model in models:
        row = sub.loc[model]
        cells = []
        for metric in metrics:
            val = row[metric]
            txt = f"{val:.3f}"
            if val == best_vals[metric]:
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        lines.append(f"& {model} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")
    
    # relative improvements of SASRec+ vs best per metric
    base_plus = sub.loc['SASRec+']
    rel_plus = []
    for metric in metrics:
        best = best_vals[metric]
        base_val = base_plus[metric] if base_plus[metric] != 0 else 1e-9
        rel = (best - base_val) / base_val * 100
        rel_plus.append(f"+{rel:.0f}\\%")
    lines.append(f"& SASRec+ vs best  & " + " & ".join(rel_plus) + r" \\")
    
    # relative improvements of SASRec vs best per metric (use original SASRec row or zero if missing)
    sas_base = group.set_index('model').reindex(['SASRec']).fillna(0).loc['SASRec']
    rel_sas = []
    for metric in metrics:
        best = best_vals[metric]
        base_val = sas_base[metric] if sas_base[metric] != 0 else 1e-9
        rel = (best - base_val) / base_val * 100
        rel_sas.append(f"+{rel:.0f}\\%")
    lines.append(f"& SASRec vs best  & " + " & ".join(rel_sas) + r" \\")
    
    lines.append(r"\midrule")

# End table
lines += [r"\end{tabular}",r"\label{tab:sasrec_plus_results}", r"\end{table}"]

# Write to .tex
with open('../../hypsasrec/Hyperbolic-SASRec/table2.tex', 'w') as f:
    f.write("\n".join(lines))

print("Table 2 LaTeX saved to table2.tex")