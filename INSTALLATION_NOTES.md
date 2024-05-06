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
1. Install the dependencies with `poetry install --with notebooks`. For windows users with Intel CPUs, please use `poetry install --with notebooks --with intel`
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


### 1.4 Test notes

The installation instructions and tutorial notebooks have been tested in the following environments:
* windows + intel
* linux + amd