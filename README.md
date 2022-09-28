
<h2 align="center">
  <br>
  <a><img src="docs\plot.png" alt="plot" width="800"></a>
  <br>
  Comparison of Uncertainty Quantification Methods in DL Concerning Resource Limited Applications
  <br>
</h2>

<h4 align="center"> Repository of a thesis for the comparison of Laplace Approximation and Multi-Input Multi-Output neural network models for systems that have limited computational resources. The final version of the thesis can be obtained from <a href="docs\thesis_final.pdf" target="_blank">here</a>.</h4>



<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#structure-of-the-repository">Repository Structure</a> •
  <a href="#functionalities">Functionalities</a> •
  <a href="#issues">Issues</a> •
  <a href="#to-do">To do</a> •
  <a href="#license">License</a>
</p>



## Abstract

The ability of deep learning methods to create models that can recognize
complex patterns with a relatively small amount of effort quickly led to their
adoption in many real-world applications. However, the inability of these
models to provide accurate uncertainty estimation regarding their results has
hampered the many possible use cases for these models and quantifying the
uncertainty of their predictions has received many possible solutions from the
academic world. This thesis investigated two such methods, namely Laplace
Approximation and Multi Input Multi Output (MIMO) ensemble, in their ability to
provide meaningful uncertainty estimation and create accurate predictions with
near certain confidence under resource limitations. During this investigation,
MIMO models have consistently outperformed the models with Laplace
Approximation but have proven to be much harder to get a viable working
model. Because of this, the ease of application for the Laplace Approximation
method was still found to be an appealing solution when practicality is a serious concern.

## Structure of the Repository
The main folder of the repository contains the python scripts and Jupyter notebooks that were used to create base models, MIMO models and the Laplace approximation models using the base models. The Models folder contains the models that were used while obtaining the results presented in the thesis. The utils folder mainly consists of the classes and functions for benchmarking, evaluating and presenting the results.


## Functionalities

This repository contains code that can be of further use in the following ways:
- Examples of MIMO models for dense and convolutional neural network models that use Optuna library for hyperparameter selection. Classes for the loading, training and validating of the models that are created, in Optuna studies.
- Example use of Laplace Redux library for the Laplace approximation of conventional neural networks
- Functions that retrieve, process and make statistical analysis of the results of non-deterministic models such as Laplace models, or multiple outputs such as MIMO models, and return a structured model output dictionary with all this information.
- Functions that process the model output dictionary to create Pandas dataframe and make further analysis/filtering on these results
- Pytorch sampler which enables the equal sampling of classes for a dataset
- Benchmarking functions that calculate the flops, and parameters of all the models and calculate the runtime for each model
- Functions used for adding motion blur and Gaussian blur to the images in the casting dataset for the addition of artifacts to the dataset instances.


## Issues

- Some Laplace models that were used in the thesis were too big to include in the Github repository. They can be recreated by using the Laplace approximation scripts.
- Casting datasets (both original and modified) were included as a zip file and the use of these datasets in the scripts requires the unzipping of related zip files
- The code, in general, is not well commented, the parts of this repository that would be useful by themselves will be shared further as standalone projects or Gists and only then will be extensively commented


## To do

- Create a separate repository for MIMO convolutional models created with Optuna
- Create Gist files for helper functions that would have further use
- Refactor the file structure in order to make it easily understandable to use
- Add Anaconda environment file


## License

MIT