if (!require("simplePHENOTYPES")) install.packages("simplePHENOTYPES", repos='http://cran.us.r-project.org')
library(simplePHENOTYPES)

# This code is used from verification.py to simulate quantitative traits
# by using simplePHENOTYPES.

# This code loads the vcf file and simulates quantitative traits

# The commandline input has 6 elements
# [num_causal, num_trait, add_effect, add_effect_2, directory_name,
# random_seed]
myArgs <- commandArgs(trailingOnly = TRUE)

num_causal <- as.numeric(myArgs[1])
num_trait <- as.numeric(myArgs[2])
add_effect <- as.numeric(myArgs[3])
add_effect_2 <- as.numeric(myArgs[4])
directory_name <- myArgs[5]
random_seed <- as.numeric(myArgs[6])

if (num_trait == 1){
  effect <- add_effect
  mean <- 0
  h2 <- 1
} else if (num_trait == 2){
  effect <- c(add_effect, add_effect_2)
  mean <- c(0,0)
  h2 <- c(1,1)
}

suppressMessages(create_phenotypes(
  geno_file = paste0(directory_name, "/tree_seq.vcf"),
  add_QTN_num = num_causal,
  add_effect = effect,
  rep = 1,
  h2 = h2,
  model = "A",
  seed = random_seed,
  vary_QTN = FALSE,
  to_r = FALSE,
  sim_method = "geometric",
  quiet = TRUE,
  home_dir = directory_name,
  verbose = FALSE,
  mean = mean,
  ntraits = num_trait
))