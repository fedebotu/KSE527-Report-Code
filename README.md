# KSE527 Project Report Code Base

Authors:
- Federico Berto
- Chuanbo Hua
- Esmée Henrieke Anne de Haas
- Emeline Bagoris

This repository contains some of the code base for the KSE527 Project Report.

## Getting Started
To get started, install the required packages. With `pip`:
```shell
pip install -r requirements.txt
```

You may find under `notebooks/training/` an example of training. The part to change for testing new models is the `SurrogateDynamicModel`: instead of a "vanilla_nn" you can try out different things. Remember to log the name of your model in Wandb for easy comparisons!


## Repository Structure
The structure mainly follows the guidelines of [PyTorch Ligthning](https://www.pytorchlightning.ai/). The structure is roughly:

```
.
├── data/
│   ├── pendulum 
│   └── (your data)
├── notebook/
│   ├── architectures
│   └── (other experiments)
├── src/
│   ├── callbacks/ (logging for Wandb)
│   ├── control/ (control related utils)
│   ├── models/ (your machine learning custom models here)
│   ├── systems/ (dynamical systems)
│   ├── tasks/ (PyTorch Lightning Learners, you may add your own)
│   └── utils/ (data generation, data modules etc)
```
## Experiments 
You may find experiments by category under the `notebooks/` folder.
Jupyter notebooks should be self-contained, while scripts can be run for example by:

```shell
python train.py
```
and can be tracked via [Wandb](https://wandb.ai/).

## Data generation
To generate the pendulum data you may run the following:
```shell
python src/utils/generate_data.py
```
from the main folder. If `src not found` happens, you may do `export PYTHONPATH = '.'` or add `import sys; sys.path.append('../../')` (as many `../` as the folders you have to go back) in the script you want to run.

You can modify the `generate_data.py` to try out another system as well!

## Feedback

Feel free to contact us in case there were any bugs or questions!