# NPMH and NPLiftedMH

We compare our Turing implementation of the NPMH and NPLiftedMH samplers with Turing's built-in SMC sampler. Note that even through we have added a seed, Turing’s SMC implementation is not deterministic.

## Getting started

1. Download and install Julia by following the instructions at [https://julialang.org/downloads/](https://julialang.org/downloads/).
2. Run `julia` from the command line to start a Julia interactive session (also known as a read-eval-print loop or "REPL").
3. Run `] add Turing, Random, Distributions, DataFrames, CSV, PlotlyJS` to install essential Julia packages for our implementation.

## Generating Samples using the SMC, NPMH and NPLiftedMH Samplers

_Note that Turing's SMC implementation is nondeterministic, so its results may vary somewhat._

1. Run `] activate NPMH` on the REPL to activate the NPMH package.
2. Run `include("infinite_gmm_npmh.jl")` on the REPL to sample from the infinite Gaussian mixture model using the SMC and NPMH samplers and store them in the data folder.
3. Run `] activate NPLiftedMH` on the REPL to activate the NPLiftedMH package.
4. Run `include("infinite_gmm_npmhp.jl")` on the REPL to sample from the infinite Gaussian mixture model using the NPLiftedMH sampler and store them in the data folder.

## Visualising the Samples

1. Run `include("visualise.jl")` on the REPL to plot the histogram of the posterior and store it in the images folder.
