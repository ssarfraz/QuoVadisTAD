# Multivariate Time Series Anomaly Detection
See the following three papers for reference on the problem, associated datasets and evaluation.

1. [Graph Deviation Network (GDN) - Graph Neural Network-Based Anomaly Detection in Multivariate Time Series](https://arxiv.org/pdf/2106.06947.pdf)
2. [TranAD - Deep Transformer Networks for Anomaly Detection in
Multivariate Time Series Data](https://arxiv.org/pdf/2201.07284.pdf)
3. [Towards a Rigorous Evaluation of Time-series Anomaly Detection](https://arxiv.org/pdf/2109.05257.pdf)


## Notebooks

 Notebook to get started. Loading a dataset and evaluation.

* [Dataset loading and Evaluation](./notebooks/dataset_loading_and_evaluation.ipynb)

* [Final best results of full evaluations of our simple methods.](./notebooks/Baselines_Evaluation_v2.ipynb)

* [Final best results of full evaluations of our simple neural networks.](./notebooks/SOTA_Evaluation.ipynb)

* [See other various implemented models and their training and testing. (Python Version <3.10.0)](./notebooks/models_train_test.ipynb)


### Important: Running Notebooks in Jupyter Labs:

1. `cd` to the project root (or start a terminal in that directory)
2. run `poetry install --with notebooks` (has to be done only once) to install the required package for running jupyter
3. run `poetry run jupyter lab` to open jupyter labs

> When not running the notebook kernel from the poetry env, make sure to install the package and dependencies in the kernel env manually or install package from Pypi (**not recommended** as it might not be the same version as yours).

Alternative for non-poetry users (not recommended):
1. `cd` to the project root (or start a terminal in that directory)
2. run `pip install .[notebooks]` (has to be done only once) to install the required package for running jupyter
3. run `python -m jupyter lab` to open jupyter labs

## Notes

* We have all five datasets i.e. WADI, SWaT, SMD, MSL and SMAP. in this repo only MSL is included because of the size limits. Other datasets are placed somewhere on the local network. please copy those into the [processed_datasets] folder.
* TranAD repo is already included. Its [main](TranAD/main.py) function can be used to train their + some other approaches on any of the datasets.
* We should now try out some of our ideas ...  

## Documentation
The generated documentation can be found in the `doc` folder.

More information on how to document the code can be found [here](https://pdoc.dev/docs/pdoc.html#how-can-i).

## Contributing
1. create issue to resolve
2. create branch following the naming convention `#[issue_number]_[branch_name]`
3. clone repository
4. if not already installed, [install poetry](https://python-poetry.org/docs/#installation)
5. to install the required dependencies run `poetry install --with dev`
7. Make your changes. To check your code and update the documentation, run `poetry run nox` and check for errors / failures.
8. [Add](https://python-poetry.org/docs/cli/#add) new or [remove](https://python-poetry.org/docs/cli/#remove) old dependencies using [poetry](https://python-poetry.org/docs/). 
9. run `poetry run nox`
   > This will automatically format the code and run all tests
10. Commit your changes.
11. create pull request and wait for review
