# Comparison

This folder includes files and folders that are necessary for comparing tstrait with the existing simulators.

R must be installed on your computer to run the codes.

- alphasim.R - This R code simulates quantitative trait based on a tree sequence in tmp folder.
- tmp - This is a folder that stores the simulated tree sequence data. Even though it is empty, please do not delete the folder, as it will not produce simulation results.

Here, we are scaling the tstrait's simulated genetic values to be consistent with the AlphaSimR's simulations, so altering tstrait's dependence on num_causal for mean and variance will not change the QQ-plot.

This test is used to make sure that tstrait is accurately simulating effect sizes from a normal distribution and also to make sure that tstrait's simulated genetic values have the desired properties.

Phenotype/environmental noise are not tested against AlphaSimR, as AlphaSimR is using a different algorithm to simulate genetic values (they are scaling genetic values before simulating environmental noise), and we have written sufficient tests to make sure that environmental noise is following a desired distribution.