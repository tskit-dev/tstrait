if (!require("AlphaSimR")) install.packages("AlphaSimR", repos='http://cran.us.r-project.org')
library(AlphaSimR)
if (!require("reticulate")) install.packages("reticulate", repos='http://cran.us.r-project.org')
library(reticulate)

# This code is used from verification.py to simulate quantitative traits
# by using AlphaSimR.
#
# The basic simulation step is the following:
# 1. Simulate tree sequence by using msprime
# 2. Use the tskit Python package through the R package tskit and load it
#   as a founder population in AlphaSimR
#   The codes of this step are largely adapted from
#   https://github.com/gaynorr/AlphaSimR_Examples/blob/master/misc/msprime.R
# 3. Simulate quantitative traits of the founder population in AlphaSimR
#  This can be achieved by setting additive trait through addTraitA
#  and extract its genetic value information through gv and phenotype
#  information through pheno
# Note: Please refer to the AlphaSimR documentation for the details
# of the simulation code written for AlphaSimR

# The commandline input has 5 elements
# [mean, var, h2, num_causal, name]
myArgs <- commandArgs(trailingOnly = TRUE)
# Convert to numerics
arguments = as.numeric(myArgs)
mean = arguments[1]
var = arguments[2]
h2 = arguments[3]
num_causal = arguments[4]
# Use this name argument to be consistent with the tree sequence file used
# for simulation
name = arguments[5]
# Use this argument to extract the individual ID
# Note: This ID is in Python, so we should add 1 to it to make it as
# a R ID
ind_id = arguments[6] + 1

tskit = import("tskit")
filename = paste0("data/tmp/tree_sequence_AlphaSim_",
              name,".trees")
ts = tskit$load(filename)
sites = ts$tables$sites$asdict()
pos = sites$position * 1e-8 # Convert to Morgans
pos = pos - pos[1] # Set first position to zero

# Extract haplotypes
haplo = t(ts$genotype_matrix())

# Create an AlphaSimR founder population
founderPop = newMapPop(genMap=list(pos), haplotypes=list(haplo))

num_rep = 200
genetic = numeric(num_rep)

for (i in 1:num_rep) {
  # Setting the parameters of the simulation
  # Quantitative trait simulation is performed here when we set the
  # parameters
  SP = SimParam$
    new(founderPop)$
    addTraitA(
      nQtlPerChr=num_causal,
      mean=mean,
      var=var
    )$
    setVarE(H2=h2)

    individuals = newPop(founderPop)
    genetic[i] = individuals@gv[ind_id, 1]
  }

df = data.frame("genetic" = genetic)
filename = paste0("data/alphasim_result/sim_",name,".csv")
write.csv(df, filename, row.names=FALSE)
