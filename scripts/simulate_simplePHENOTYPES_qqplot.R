if (!require("simplePHENOTYPES")) install.packages("simplePHENOTYPES", repos='http://cran.us.r-project.org')
library(simplePHENOTYPES)

# This code is used from verification.py to simulate quantitative traits
# by using simplePHENOTYPES.

# This code loads the vcf file that is generated in `verification.py`
# and uses the effect size from a normal distribution to simulate
# additive traits.

# The commandline input has 7 elements
# [num_causal, h2, directory_name,
# num_rep, mean, var, random_seed]
myArgs <- commandArgs(trailingOnly = TRUE)

num_causal <- as.numeric(myArgs[1])
h2 <- as.numeric(myArgs[2])
directory_name <- myArgs[3]
num_rep <- myArgs[4]
mean <- as.numeric(myArgs[5])
var <- as.numeric(myArgs[6])
random_seed <- as.numeric(myArgs[7])

set.seed(random_seed)

sd <- sqrt(var)

# Function to simulate phenotypes from simplePHENOTYPES
# The effect sizes are simulated from a normal distribution,
# as the geometric series is the only effect size distribution
# supported in simplePHENOTYPES.
simulate_simplePHENOTYPE <- function(
    num_causal, random_seed
    ) {
  effect_size <- list(rnorm(n=num_causal, mean=mean, sd=sd))
  phenotypes <- suppressMessages(create_phenotypes(
    geno_file = paste0(directory_name, "/tree_seq.vcf"),
    add_QTN_num = num_causal,
    add_effect = effect_size,
    rep = 1,
    h2 = h2,
    model = "A",
    seed = random_seed,
    vary_QTN = FALSE,
    to_r = TRUE,
    sim_method = "custom",
    quiet = TRUE,
    home_dir = directory_name,
    verbose = FALSE,
    mean = 0
  ))
  # The mean is centered at 0 from simplePHENOTYPES simulation
  # so we will divide it by the standard deviation to normalise
  # the data
  phenotypes[,2] <- phenotypes[,2] / sd(phenotypes[,2])
  names(phenotypes)[1:2] <- c("individual_id", "phenotype")
  return(phenotypes)
}

phenotype_result <- c()

for (i in 1:num_rep) {
  simulated_result <- simulate_simplePHENOTYPE(
    num_causal=num_causal, random_seed=random_seed+i
  )
  phenotype_result <- rbind(phenotype_result, simulated_result)
}

filename = paste0(directory_name, "/simplePHENOTYPES.csv")
write.csv(phenotype_result, filename, row.names=FALSE)
