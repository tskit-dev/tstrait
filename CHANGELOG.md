# Changelog

## [0.1.0] - 2024-03-07

### Breaking changes:

- `sim_genetic` function is no longer supported, and users should be using `genetic_value` function instead. The new `genetic_value` function uses trait dataframe as an input, but `random_seed` is not a necessary argument, as there is no randomness involved. The frequency dependence architecture is implemented in `sim_trait` function instead of `sim_genetic` function, so users should put the `alpha` parameter in `sim_trait` function instead.
- `negative` input in exponential and gamma distribution trait models are no longer supported, and users should be using `random_sign` instead {pr}`114`

### Update:

- Remove `num_causal` dependence on simulating effect sizes {pr}`107`
- Add options to simulate effect sizes from `random_sign` in fixed value trait model {pr}`109`
- Add frequency dependence architecture in `sim_trait` function, and allele frequency is given as an output as well {pr}`111`
- Implement `genetic_value` function to compute genetic values based on the trait dataframe. There is no randomness involved, and frequency dependence architecture is not implemented {pr}`112`
- Add option to input the causal site IDs, instead of randomly selecting them in `sim_phenotype` and `sim_trait` functions {pr}`124`
- Add `normalise_phenotypes` function to normalize the simulated phenotypes {pr}`130`
- Add delta degrees of freedom input in `normalise_phenotypes` function {pr}`136`
- Add `normalise_genetic_value` function to normalize the genetic values {pr}`145`

### Fix:

- Raise error when there are no individuals {pr}`97`
- Raise error when incorrect values are given in the `num_causal` argument {pr}`99`
- Remove `# pragma: no cover` in certain functions {pr}`119`
- Modify default input arguments of `sim_trait`, `sim_env` and `sim_phenotype` functions {pr}`120`
- Add `verification.py` for statistical tests {pr}`129`
- Add statistical tests against external simulators {pr}`132`
- Change the dtype of `trait_id` input in `genetic_value` function {pr}`134`
- Add density plot in `verification.py` {pr}`138`
- Add multithreading in `verification.py` {pr}`139`
- Conduct exact tests against AlphaSimR, simplePHENOTYPES and the simulation framework described in ARG-Needle paper in `verification.py` {pr}`140`

### Documentation:

- Modify introduction {pr}`96`
- Document ploidy {pr}`98`
- Documentation for the new `sim_trait` function {pr}`115`
- Documentation for `random_sign` input in trait distribution models {pr}`122`
- Modify phrasing in documentation {pr}`123`
- Documentation for specifying causal site IDs {pr}`126`
- Documentation for modifying the numericalization of genotypes {pr}`133`
- Modify the frequency dependence explanation in the documentation {pr}`141`
- Fix typo in documentation {pr}`142`


## [0.0.1] - 2023-09-05

### Highlights:

- Initial stable release of tstrait in PyPI https://pypi.org/project/tstrait/
- Initial stable release of tstrait in conda-forge 

### Documentation:

- tstrait description in https://tskit.dev/software/

### Contributors:

- Jerome Kelleher
- Ben Jeffery
- Gertjan Bisschop
- Daiki Tagami

## [0.0.1a5] - 2023-09-05

Test release of the package before releasing it to conda-forge

### Contributors:

- Jerome Kelleher
- Ben Jeffery
- Gertjan Bisschop
- Daiki Tagami

## [0.0.1a2] - 2023-08-25

### Highlights:

- Release of tstrait documentation in https://tskit.dev/tstrait/docs/latest/

### Fix:

- Hide private functions and classes {pr}`73`
- Modify docstring explanations and examples {pr}`76`

### Documentation:

- Create infrastructure for documentation {pr}`77`
- Build initial documentation {pr}`78`
- Add Changelog to documentation {pr}`79`

### Contributors:

- Daiki Tagami
- Gertjan Bisschop
- Jerome Kelleher

## [0.0.1a1] - 2023-08-22

Initial alpha release of the package.

### Contributors:

- Daiki Tagami
- Gertjan Bisschop
- Jerome Kelleher
