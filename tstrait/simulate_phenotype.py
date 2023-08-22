from dataclasses import dataclass

import pandas as pd
import tstrait


@dataclass
class PhenotypeResult:
    """
    Dataclass that contains effect size dataframe and phenotype dataframe.

    Attributes
    ----------
    effect_size : pandas.DataFrame
        Dataframe that includes simulated effect sizes.
    phenotype : pandas.DataFrame
        Dataframe that includes simulated phenotype.

    See Also
    --------
    sim_phenotype : Use this dataclass as a simulation output.

    Examples
    --------
    See :ref:`effect_size_output` for details on extracting the effect size
    dataframe, and :ref:`phenotype_output` for details on extracting the
    phenotype dataframe.
    """

    effect_size: pd.DataFrame
    phenotype: pd.DataFrame


def sim_phenotype(ts, num_causal, model, h2, alpha=0, random_seed=None):
    """
    Simulate quantitative traits.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence data that will be used in the quantitative trait
        simulation.
    num_causal : int
        Number of causal sites.
    model : tstrait.TraitModel
        Trait model that will be used to simulate effect sizes.
    h2 : float or array-like
        Narrow-sense heritability. When it is 0, environmental noise will be a vector of
        zeros. The dimension of `h2` must match the number of traits to be simulated.
    alpha : float, default 0
        Parameter that determines the degree of the frequency dependence model. Please
        see :ref:`frequency_dependence` for details on how this parameter influences
        effect size simulation.
    random_seed : int, default None
        Random seed of simulation. If None, simulation will be conducted randomly.

    Returns
    -------
    PhenotypeResult
        Dataclass object that includes phenotype and effect size dataframe.

    Raises
    ------
    ValueError
        If the number of mutations in `ts` is smaller than `num_causal`.
    ValueError
        If `h2` <= 0 or `h2` > 1

    See Also
    --------
    trait_model : Return a trait model, which can be used as `model` input.
    PhenotypeResult : Dataclass object that will be used as an output.

    Notes
    -----
    The simulation outputs of effect sizes and phenotypes are given as a
    :py:class:`pandas.DataFrame`.

    The effect size dataframe can be extracted by using ``.effect_size`` in the
    resulting object and contains the following columns:

        * **site_id**: ID of sites that have causal mutation
        * **effect_size**: Genetic effect size of causal mutation
        * **trait_id**: Trait ID and will be used in multi-trait simulation.
        * **causal_state**: Causal state.
        * **allele_frequency**: Allele frequency of causal mutation.

    The phenotype dataframe can be extracted by using ``.phenotype`` in the resulting
    object and contains the following columns:

        * **trait_id**: Trait ID.
        * **individual_id**: Individual ID inside the tree sequence input.
        * **genetic_value**: Simulated genetic values.
        * **environmental_noise**: Simulated environmental noise.
        * **phenotype**: Simulated phenotype.

    Please refer to :ref:`phenotype_model` for mathematical details of the phenotypic
    model.

    Examples
    --------
    See :ref:`quickstart` for worked examples.

    """
    trait_df = tstrait.sim_trait(
        ts=ts, num_causal=num_causal, model=model, random_seed=random_seed
    )
    genetic_result = tstrait.sim_genetic(
        ts=ts, trait_df=trait_df, alpha=alpha, random_seed=random_seed
    )
    phenotype_df = tstrait.sim_env(
        genetic_df=genetic_result.genetic, h2=h2, random_seed=random_seed
    )

    result = tstrait.PhenotypeResult(
        effect_size=genetic_result.effect_size, phenotype=phenotype_df
    )

    return result
