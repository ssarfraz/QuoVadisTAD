# Multivariate Time Series Anomaly Detection
See the following three papers for reference on the problem, associated datasets and evaluation.

1. [Graph Deviation Network (GDN) - Graph Neural Network-Based Anomaly Detection in Multivariate Time Series](https://arxiv.org/pdf/2106.06947.pdf)
2. [TranAD - Deep Transformer Networks for Anomaly Detection in
Multivariate Time Series Data](https://arxiv.org/pdf/2201.07284.pdf)
3. [Towards a Rigorous Evaluation of Time-series Anomaly Detection](https://arxiv.org/pdf/2109.05257.pdf)


## Notebooks

 Notebook to get started. Loading a dataset and evaluation.

* [Dataset loading and Evaluation](./notebooks/dataset_loading_and_evaluation.ipynb)

* [Final best results of full evaluations of our simple methods.](./notebooks/Baelines_Evaluation.ipynb)

* [See other various implemented models and their training and testing.](./notebooks/models_train_test.ipynb)


## Notes

* We have all five datasets i.e. WADI, SWaT, SMD, MSL and SMAP. in this repo only MSL is included because of the size limits. Other datasets are placed somewhere on the local network. please copy those into the [processed_datasets] folder.
* TranAD repo is already included. Its [main](TranAD/main.py) function can be used to train their + some other approaches on any of the datasets.
* We should now try out some of our ideas ...  