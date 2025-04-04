<h2 align="center">partition-nf-expansion: Approximating non-Gaussian Bayesian partitions with normalising flows</h2>

<p align="center">
<a href="https://arxiv.org/abs/2501.04791"><img alt="Arxiv" src="https://img.shields.io/badge/arXiv-2407.06259-b31b1b.svg"></a>

In this repository, we provide our code used to obtain the results from our paper on approximating non-Gaussian partitions with normalising flows. The implementation of the entropy calculation via the normalising flow and of the flow expansion approximation are found in the "flow_expansion" folder. An exemplary application on a Gaussian toy model is shown in "example.ipynb".

Replacing the "samples" in this notebook allows to train the network on one's own posterior data. In that manner, one could for example manually include the [Union2.1][Union2.1] dataset (after obtaining a posterior via the [emcee][emcee] package for example).

We want to mention again, that our implementation is based on the normalising flow package [FrEIA][FrEIA].

[Union2.1]: https://supernova.lbl.gov/union/figures/SCPUnion2.1_mu_vs_z.txt
[emcee]: https://emcee.readthedocs.io/en/stable/
[FrEIA]: https://github.com/vislearn/FrEIA

## Required packages

To successfully run the example notebook, the following packages are required:
- [PyTorch][PyTorch]
- [NumPy][NumPy]
- [SciPy][SciPy]
- [Matplotlib][Matplotlib]
- [pandas][pandas]
- [seaborn][seaborn]
- [FrEIA][FrEIA] (which can easily be installed via "pip install FrEIA" e.g.)

[pandas]: https://pandas.pydata.org
[seaborn]: https://seaborn.pydata.org
[Matplotlib]: https://matplotlib.org
[SciPy]: https://scipy.org
[PyTorch]: https://pytorch.org
[NumPy]: https://numpy.org

## Usage
To use the code, simply clone this repository:
```sh
# clone the repository
git clone https://github.com/cosmostatistics/partition-nf-expansion.git
```
In order to execute the "example.ipynb", make sure that you have the Jupyter kernel installed.

## Acknowledgements

If you use any part of this repository please cite the following paper:

```
@misc{schosser2024markovwalkexplorationmodel,
      title={Approximating non-Gaussian Bayesian partitions with normalising flows: statistics, inference and application to cosmology}, 
      author={R{\"o}spel, Tobias and Schlosser, Adrian and Sch{\"a}fer, Bj{\"o}rn Malte},
      year={2025},
      eprint={2501.04791},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO},
      url={https://arxiv.org/abs/2501.04791}, 
}
```