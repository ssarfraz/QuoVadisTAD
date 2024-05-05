# Quo Vadis, Unsupervised Time Series Anomaly Detection?

QuoVadisTAD contains the artifacts of our ICML 2024 position paper [Quo Vadis, Unsupervised Time Series Anomaly Detection?]().  

Our position paper scrutinizes the prevailing practices in Time Series Anomaly Detection (TAD), pinpointing issues with persistent use of flawed evaluation metrics, benchmarking inconsistencies, and unnecessary complexity in the offered deep learning based models for the task. We advocate for consistant benchmarking, better datasets, and a focus on model utility by drawing simpler baselines. Our findings suggest that simple methods often exceed the performance of complex TAD deep models, which typically learn linear mappings. We offer simple & effective baselines and a repository with benchmarks and strict evaluation protocols to redirect TAD research towards more meaningful progress.

Although this project started as academic research, feedbacks from practitioners show the potential of further developing it with tools that prioritize clarity and effectiveness and make it a valuable asset for the academic research and open-source communities. 

If you would be interested to contribute, please follow the `how to contribute` section below.



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

To manage the dependencies we use [Poetry](https://python-poetry.org/), though non-poetry users can also easily install. For detailed installation instructions, please see the [Installation notes](./installation_notes.md).

## 2. Overview

### 2.1 Notebooks tutorials

Two notebooks are provided as `quick-start` tutorials for running and evaluating our simple & neural network baselines on any dataset/s.

* [Run Simple baselines on dataset/s and evaluate](./notebooks/Simple_Baselines_Evaluation.ipynb)

* [Train and test Neural Network baselines](./notebooks/NN_Baselines_models_train_test.ipynb)


### 2.2 Datasets

We use various multivariate and univariate datasets, see the paper for details. For the purpose of consistant benchmarking and evaluation, we provide the prepared datasets downloadable from this [Download Link](https://bwsyncandshare.kit.edu/s/YixoyiHx2xf8yZ9). Please note some datasets such WADI and SWaT still requires to signup an access request on the providers website, see the [readme](./resources/processed_datasets/readme.md) in the processed_datasets folder for respective copyright notices and details. After downloading the datasets please copy those into the [./resources/processed_datasets] folder.

 
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
