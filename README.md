# Quo Vadis, Unsupervised Time Series Anomaly Detection?

QuoVadisTAD contains the artifacts of our position paper published at the 2024 International Conference on Machine Learning (ICML): [Quo Vadis, Unsupervised Time Series Anomaly Detection?](). We advocate for a shift towards rigorous benchmarking, rich datasets, and a nuanced understanding of model utility. Our findings debunk the myth that complexity equals superiority, showcasing the prowess of simpler, more interpretable methods. This repository aims to faciliate the course of correction by offering meticulously prepared datasets, simple baselines, and rigorious evaluation protocols. 

Although this project began in pursue of academic research, feedbacks from practitioners show the potential of further developing it into a Free and Open Source Software (FOSS). If you would be interested, please follow the `how to contribute` section below. Let's shape the future of **time series anomaly detection (TAD)** with tools that prioritize clarity and effectiveness and make it an invaluable asset for the academic research and open-source communities. 

**Citation**

If you use the `QuoVadisTAD` components in your research, please cite our paper via:

```
@inproceedings{quovadisTAD,
  title={Position Paper: Quo Vadis, Unsupervised Time Series Anomaly Detection?},
  author={M. Saquib Sarfraz, Mei-Yen Chen, Lukas Layer, Kunyu Peng, Marios Koulakis},
  booktitle={International Conference on Machine Learning},
  pages={},
  year={2024},
}
```


## 1. Installation

The applications of the simple baselines and evaluation protocols are presented in jupyter notebooks. We use [Poetry](https://python-poetry.org/) to manage the dependencies for the demo notebooks. If you were not familiar with Poetry, please follow the [1.1 Poetry Installation](). 


**Alternative for non-poetry users**
1. `cd` to the project root (or start a terminal in that directory)
1. run `pip install .[notebooks]` (has to be done only once) to install the required package for running jupyter
1. run `python -m jupyter lab` to open jupyter labs


### 1.1 Poetry Installation

Depending on your OS system, we recommend the following stepts to install Poetry:

#### Linux and MacOS
1. Install `pyenv` using the instructions [here](https://github.com/pyenv/pyenv#installation).
2. Install `poetry`:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Install the required Python versions (we recommend at least 3.10) using `pyenv` and activate it for the example folders:
   ```bash
   pyenv install 3.10.0
   cd QuoVadisTAD
   pyenv local 3.10.0
   ```
Now you can use the `poetry` command to continue the installation of this package and execute the tutorial notebooks.

#### Windows
Managing multiple Python versions on Windows can be a bit tricky. We recommend using `conda` for this purpose, which you can install using the installer from [here](https://docs.conda.io/en/latest/miniconda.html). After installing conda, you can create a new conda environment with Python 3.10 (or any other python version) and install Poetry into it using the following commands:
```bash
conda create -n poetry_1_7_py_3_10 python=3.10
conda activate poetry_1_7_py_3_10
pip install poetry~=1.7 poetry-conda
```

Repeat the above instructions for any other Python version you may want to use. For running the example notebooks, we recommend using at least Python 3.10.
To use Poetry, simply activate the environment it was installed into:
```bash
conda activate poetry_1_7_py_3_10
```
Now you can use the `poetry` command to continue the installation of this package and execute the tutorial notebooks.


### 1.2 Install Dependencies and execute the tutorial notebooks

To install the dependencies for running the tutorial notebooks in the Jupyter WebUI:
1. Install the dependencies using `poetry install --with notebooks`
1. Execute the tutorial notebooks in the Jupyter WebUI `poetry run jupyter lab`

Now, you should see that the Jupyter WebUI opens in a browser. On the left side, you can double-click on the notebook file that you would like to run. Jupyter WebUI offers a menu bar for you to execute the code inside the notebook's cells.


### 1.3 Install extras
The following extras are available for this package:
| extra string | description |
| --- | --- |
| dev | dependencies for deployment (like build tools) |
| notebooks | dependencies for running local jupyter server |
| intel | dependencies for increasing computation speed (exclusive to Intel CPUs) |

Extras can be installed using `poetry install --with <extra string>`, multiple extras can be installed at the same time. For non-poetry-users, the same can be done with `pip install .[<extra string>]`.


## 2. Tutorial notebooks

### 2.1 Overview - tutorials

Several notebooks have be prepared as `quick-start` tutorials for loading a dataset, fitting simple baselines, and evaluation.

* [Dataset loading and Evaluation](./notebooks/dataset_loading_and_evaluation.ipynb)

* [Final best results of full evaluations of our simple methods.](./notebooks/Baselines_Evaluation_v2.ipynb)

* [Final best results of full evaluations of our simple neural networks.](./notebooks/SOTA_Evaluation.ipynb)

* [See other various implemented models and their training and testing. (Python Version <3.10.0)](./notebooks/models_train_test.ipynb)


### 2.2 Datasets

We have all five datasets i.e. WADI, SWaT, SMD, MSL and SMAP. in this repo only MSL is included because of the size limits. Other datasets are placed somewhere on the local network. please copy those into the [processed_datasets] folder.

 
### 2.3 API Documentation

In the `doc` folder, you can find the documentation about the APIs in this package. Those documents were automatically generated by running `poetry run nox` under the project folder. More information on how to document the code can be found [here](https://pdoc.dev/docs/pdoc.html#how-can-i).


## 3. How to contribute

We welcome contribution from Free and Open Source Software community. Please follow the contribution guideline below:

1. create issue to resolve
2. create branch following the naming convention `#[issue_number]_[branch_name]`
3. clone repository using `git clone <repo-url> --recurse-submodules`
4. if not already installed, [install poetry](https://python-poetry.org/docs/#installation)
5. to install the required dependencies run `poetry install --with dev`
7. Make your changes. To check your code and update the documentation, run `poetry run nox` and check for errors / failures.
8. [Add](https://python-poetry.org/docs/cli/#add) new or [remove](https://python-poetry.org/docs/cli/#remove) old dependencies using [poetry](https://python-poetry.org/docs/). 
9. run `poetry run nox`
   > This will automatically format the code and run all tests
10. Commit your changes.
11. create pull request and wait for review
