# MultiAlternativeDecisions
Code for Marshall, Reina and Pirrone (2021) Magnitude-sensitive reaction times reveal non-linear time costs in multi-alternative decision-making.
* [Preprint](https://www.biorxiv.org/content/10.1101/2021.05.05.442775v1)

These codes modify the analysis presented in the paper by Tajima et al. (2019), to examine the implication of maximising geometric-discounted future rewards on the optimal policy, rather than maximising arithmetic mean reward rate.

In addition, numerical code is presented which simulates optimal Bayesian decision-makers using the derived optimal policies, to measure realised reaction times.


## Installation

All scripts are written in MATLAB and have been tested with MATLAB R2018b. They should also run on earlier versions of MATLAB.

To run the scripts, [download them](https://github.com/DrugowitschLab/MultiAlternativeDecisions/archive/master.zip), extract them into a folder of your choice, and navigate to this folder within MATLAB.

## Usage

To use any of the script in the `fig` folders, you need to first point MATLAB to the shared utility functions by calling
```Matlab
addpath('shared/')
addpath('shared/bads-1.0.4/')
```
at the MATLAB command line. See the individual `fig` folders for further usage instructions.

To switch between maxisation of geometrically-discounted rewards and arithmetic mean reward rate, toggle the value of the `geometric` variable.

To scale option values change the value of the `valscale` variable.

To change between linear and various nonlinear utility functions uncomment the appropriate `utilityFunc` definition.

## Usage of non-linear-time experiments

First, you must compute the decision boundaries over time. You can do that by running the `non-linear-time/computeProjections.m` script which will save the results to files stored in the `rawData/` folder.
Use the command `computeProjections("all")` to compute the boundaries for all utility functions. Otherwise, specify the desired utility function as argument, e.g., `computeProjections("linear")`.
You can add new utility functions by adding a new keyword and function in the initial lines (53~60) of the file `non-linear-time/computeProjections.m`. 
Remember to compute the boundaries of both linear and non-linear temporal discount by varying the variable `geometric = ...;`

Second, you must run the numerical simulation of the DDM. You can do that by running the `non-linear-time/numericalSimulations.m` script which will save the results to files stored in the `resultsData/` folder.

Both `computeProjections.m` and `numericalSimulations.m` take a number of compulsory arguments to specify the precise decision problem to be solved/simulated.

Finally, you can plot the computed results from sensitivity analyses (see _Cluster operation_ below) of the results using `non-linear-time/multiplot.m` script.

### Cluster operation

`qsub` scripts are provided to run batch parallel experiments on a cluster, in order to perform sensitivity analyses:
* `compute.sh`
* `numerical.sh`

