if (!require("AlphaSimR")) install.packages("AlphaSimR", repos='http://cran.us.r-project.org')
library(AlphaSimR)
if (!require("reticulate")) install.packages("reticulate", repos='http://cran.us.r-project.org')
library(reticulate)
if (!require("tidyr")) install.packages("tidyr", repos='http://cran.us.r-project.org')

# This code is used from verification.py to simulate quantitative traits
# by using AlphaSimR.
#
# The basic simulation step is the following:
# 1. Use the tskit Python package through the R package tskit and load the tree
#   sequence data as a founder population in AlphaSimR. The codes of this step are
#   largely adapted from
#   https://github.com/ ynorr/AlphaSimR_Examples/blob/master/misc/msprime.R
# 2. Simulate quantitative traits of the founder population in AlphaSimR

# The commandline input has 8 elements
# [num_causal, temporary_directory_name, tree_filename, phenotype_filename,
# trait_filename, corA, num_trait, random_seed]
myArgs <- commandArgs(trailingOnly = TRUE)
# Convert to numerics
num_causal <- as.numeric(myArgs[1])
directory_name <- myArgs[2]
tree_filename <- myArgs[3]
phenotype_filename <- myArgs[4]
trait_filename <- myArgs[5]
corA <- as.numeric(myArgs[6])
num_trait <- as.numeric(myArgs[7])
random_seed <- as.numeric(myArgs[8])

set.seed(random_seed)

tskit <- import("tskit")

tree_filename <- paste0(directory_name,"/", tree_filename,".tree")
ts <- tskit$load(tree_filename)

sites <- ts$tables$sites$asdict()
pos <- sites$position * 1e-8 # Convert to Morgans
pos <- pos - pos[1] # Set first position to zero

# Extract haplotypes
haplo <- t(ts$genotype_matrix())

# Create an AlphaSimR founder population
founderPop <- newMapPop(genMap=list(pos), haplotypes=list(haplo))

num_ind <- nrow(haplo) / 2

if (num_trait == 1){
  mean <- 0
  var <- 1
  corA <- NULL
  H2 <- 1
} else if (num_trait == 2){
  mean <- c(0,0)
  var <- c(1,1)
  corA <- matrix(c(1,corA,corA,1),nrow=2,ncol=2)
  H2 <- c(1,1)
}

SP <- SimParam$
  new(founderPop)$
  addTraitA(
    nQtlPerChr=num_causal,
    mean=mean,
    var=var,
    corA=corA
  )$
  setVarE(H2=H2)

individuals <- newPop(founderPop)

trait_df <- c()
phenotype_df <- c()

for (trait_id in 1:num_trait){
  qtl_site <- SP$traits[[trait_id]]@lociLoc - 1
  effect_size <- SP$traits[[trait_id]]@addEff
  trait_id_df <- data.frame(
    effect_size = effect_size,
    site_id = qtl_site,
    trait_id = rep(trait_id-1, length(effect_size))
  )
  trait_df <- rbind(trait_df, trait_id_df)
  phenotype <- individuals@pheno[,trait_id]
  phenotype_id_df <- data.frame(
    phenotype=phenotype,
    individual_id = 0:(num_ind-1),
    trait_id = rep(trait_id-1, num_ind)
  )
  phenotype_df <- rbind(phenotype_df, phenotype_id_df)
}

phenotype_filename <- paste0(directory_name,"/",phenotype_filename,".csv")
write.csv(phenotype_df, phenotype_filename, row.names=FALSE)

trait_filename <- paste0(directory_name,"/",trait_filename,".csv")
write.csv(trait_df, trait_filename, row.names=FALSE)