import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

tensor_dir = 'exp_ML1M_sasrecb_dists'
pattern = re.compile(r'^(aff|dist)_(\d+)_\d+\.pt$')
stride = 20  # Only process every N-th epoch

# collect tensors by type and epoch
data = defaultdict(lambda: defaultdict(list))
for fname in os.listdir(tensor_dir):
    m = pattern.match(fname)
    if not m:
        continue
    ttype, epoch_str = m.groups()
    epoch = int(epoch_str)
    if epoch % stride != 0:
        continue  # skip if not every N-th epoch
    tensor = torch.load(os.path.join(tensor_dir, fname))
    vals = tensor.detach().cpu().flatten().numpy()
    if ttype.endswith('_dist'):
        vals = vals[vals > 0]
    else:  # affinity
        vals = np.log(vals[vals != 0])  # log-transform affinity
    if vals.size > 0:
        data[ttype][epoch].append(vals)

# plot and summarize
for ttype in ('aff', 'dist'):
    epochs = sorted(data[ttype].keys())
    if not epochs:
        print(f"No data for {ttype}")
        continue
    box_data = []
    print(f"\n{ttype} statistics by epoch (filtered, every {stride} epochs):")
    for e in epochs:
        arr = np.concatenate(data[ttype][e])
        box_data.append(arr)
        print(f"  epoch {e:3d}: N={arr.size}  mean={arr.mean():.4f}  std={arr.std():.4f}")
    # plot
    plt.figure(figsize=(10, 5))
    plt.boxplot(box_data,
                labels=[str(e) for e in epochs],
                showfliers=False)
    plt.title(f"{ttype} per epoch (every {stride} epochs)")
    plt.xlabel("epoch")
    plt.ylabel(ttype)
    if ttype.endswith('aff'):
        plt.ylim(bottom=-10)  # for log-affinities
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(tensor_dir, f'{ttype}_boxplots_stride{stride}.png'))
    plt.close()
