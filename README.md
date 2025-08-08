
## Learning Geometry-Aware Recommender Systems with Manifold Regularization (RecSys 2025)
___

This repository contains the source code for the paper. It includes implementations of:
- **SASRec** and variants (`SASRec+`, `SASRec_man`, `SASRec+_man`)
- **HypSASRec** and variants (`HypSASRec+`)
- **NCF** and variants (`NCF`)

---

<details>
          <summary>Implementation Notes</summary>
          <p>
                This work extends and modifies the following open-source implementations:
                
- Hyperbolic-SASRec (https://github.com/AIRI-Institute/Hyperbolic-SASRec);
- NCF (https://github.com/guoyang9/NCF).

Original licenses and source code are preserved in their repositories.

</p>
</details>



## 🚀 Quick Start

### 1. Download Datasets
Run the following command to download and preprocess datasets:

``` bash
python scripts/get_data.py --dataset NAME
```

Supported datasets (```NAME```):

- ```ml1m``` (MovieLens 1M)

- ```Digital_Music```

- ```Arts_Crafts_and_Sewing```

- ```Grocery_and_Gourmet_Food```

- ```ffice_Products```

- ```Luxury_Beauty```

### 2. Organize Data

Move the downloaded data to the correct folder:
``` bash
cp -r scripts/data/raw data/raw
```

---

### 🔧 Reproducing Experiments

###  Grid Search (Pre-Completed)
We provide pre-tuned hyperparameters for manifold regularization. 
The original grid search was run via:

``` bash
python tune.py --model sasrec_manifold --dataset %dataset_name% --time_offset 0.95 \
               --config_path ./grids/sasrec_manifold.py --grid_steps 60 --dump_results
```

###  Run Models with Best Configs

Execute the following for each model and dataset combination:

``` bash
python test.py --model %model% --dataset %dataset_name% --time_offset 0.95 \
               --config_path ./grids/best/sasrec_%dataset_alias%.py --grid_steps 60 --dump_results
```


#### Model Options (`%model%`):
| Model               | Alias             |
|---------------------|-------------------|
| SASRec              | `sasrec`          |
| SASRec+             | `sasrecb`         |
| SASRec_man          | `sasrec_manifold` |
| SASRec+_man         | `sasrecb_manifold`|
| HypSASRec           | `hybsasrecb`      |
| HypSASRec+          | `hypsasrec`       |

#### Dataset Mappings:
| Dataset Name (`%dataset_name%`)   | Alias (`%dataset_alias%`) |
|-----------------------------------|---------------------------|
| `ml1m`                            | `ml1m`                    |
| `Digital_Music_5`                 | `dig`                     |
| `Arts_Crafts_and_Sewing_5`        | `arts`                    |
| `Grocery_and_Gourmet_Food_5`      | `grocery`                 |
| `Office_Products_5`               | `office`                  |
| `Luxury_Beauty_5`                 | `lux`                     |



## Citation
If you use this code, please cite our RecSys 2025 paper:

``` latex
@inproceedings{Zainulabidova2025,
  author = {Zainulabidova, Zaira and Borisova, Julia and Hvatov, Alexander},
  title = {Learning Geometry-Aware Recommender Systems with Manifold Regularization},
  booktitle = {Proceedings of the 19th ACM Conference on Recommender Systems (RecSys '25)},
  year = {2025},
  location = {Prague, Czech Republic},
  pages = {5},
  publisher = {ACM},
  address = {New York, NY, USA},
  doi = {10.1145/3705328.3759323},
  url = {https://doi.org/10.1145/3705328.3759323},
  series = {RecSys '25}
}
```

## Contacts

In case of any questions feel free to [contact us](mailto:alex_hvatov@itmo.ru) or open an issue in this repository.