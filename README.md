# MultiAlternativeDecisions
Code for comment on Tajima et al. (2019). Optimal policy for multi-alternative decisions. Nature Neuroscience.
* [Preprint](https://www.biorxiv.org/content/10.1101/2019.12.18.880872v1.abstract)

These codes modify the analysis presented in the paper by Tajima et al. (2019), to examine the implication of maximising geometric-discounted future rewards on the optimal policy, rather than maximising arithmetic mean reward rate.

**N.B.** reward rate across trials is not maximised, but set to 0; this has the effect of considering only single trial dynamics, since the cost of waiting for the next trial becomes zero. 

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
