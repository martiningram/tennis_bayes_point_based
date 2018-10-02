# Code for: "A point-based Bayesian hierarchical model for predicting the outcome of tennis matches"

This is a repository containing code for the paper. 

## Requirements

1. Python 2
2. The requirements in `requirements.txt`. 

## Getting started

1. Install the requirements listed in `requirements.txt`. You should be able to
   do this using `pip install -r requirements.txt`.
   Note: On some systems, old versions of `gcc` may cause Stan to fail to
   compile the model. In this case, we recommend the `conda` package manager,
   which installs the required dependencies. To use conda instead of pip,
   install the requirements using `conda install --file requirements.txt`.
2. An example of how to run the model can be found in `Example.ipynb`.

## Overview of the repository

The repository contains the following files:

* `Example.ipynb`, containing an example of how to run the model
* `bayes_point_model.py`, the python code implementing the model
* `dataset.csv`, the data used to fit the model (ATP match results from 2011
  onwards)
* `requirements.txt`, listing the requirements to install the model
* `stan_model.stan`, the Stan code to fit the model
* `winning_prob.py`, code calculating the iid win probability given
  serve-winning probabilities, written inpython.
