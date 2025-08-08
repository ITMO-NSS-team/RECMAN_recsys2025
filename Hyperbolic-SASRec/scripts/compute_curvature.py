import os
import argparse
import logging
from datetime import datetime

import numpy as np
from tqdm import tqdm

import libcontext
from lib import defaults, logger
from lib.delta import relative_delta_poincare
from lib.utils import dump_intermediate_results, try_tuncate, validate_list
from lib.embed import get_item_representations




import json

# Read the file
with open('data/results/delta_Luxury_Beauty_5_25-04-2025_13-55-14.txt', 'r') as file:
    data = json.load(file)  # Parse the JSON array into a Python list of dictionaries

curv = []

delta_p_rel = relative_delta_poincare(tol=1e-12)

for entry in data:
    c = (delta_p_rel/np.mean(entry['delta']))**2
    curv.append(c)



print('curvature = {} \pm {}'.format(np.mean(curv),np.std(curv)))