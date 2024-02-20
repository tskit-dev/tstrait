from dataclasses import dataclass

import numpy as np
import pandas as pd
import tstrait

from .base import _check_dataframe


@dataclass
class PhenotypeResult:
    """
    Dataclass that contains effect size dataframe and phenotype dataframe.

    Attributes
    ----------
    trait : pandas.DataFrame
        Trait dataframe that includes simulated effect sizes.
    phenotype : pandas.DataFrame
        Phenotype dataframe that includes simulated phenotype.

    See Also
    --------
    sim_phenotype : Use this dataclass as a simulation output.

    Examples
    --------
    See :ref:`effect_size_output` for details on extracting the trait
    dataframe, and :ref:`phenotype_output` for details on extracting the
    phenotype dataframe.
    """

    trait: pd.DataFrame
    phenotype: pd.DataFrame


def sim_phenotype(
    ts,
    model,
    *,
    num_causal=None,
    causal_sites=None,
    alpha=None,
    h2=None,
    random_seed=None
):
    """
    Simulate quantitative traits.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence data that will be used in the quantitative trait
        simulation.
    model : tstrait.TraitModel
        Trait model that will be used to simulate effect sizes.
    num_causal : int, default None
        Number of causal sites. If None, number of causal sites will be 1.
    causal_sites : list, default None
        List of site IDs that have causal allele. If None, causal site IDs will be
        chosen randomly according to num_causal.
    alpha : float, default None
        Parameter that determines the degree of the frequency dependence model. Please
        see :ref:`frequency_dependence` for details on how this parameter influences
        effect size simulation. If None, alpha will be 0.
    h2 : float or array-like, default None.
        Narrow-sense heritability. When it is 1, environmental noise will be a vector of
        zeros. If `h2` is array-like, the dimension of `h2` must match the number of
        traits to be simulated. If None, h2 will be 1.
    random_seed : int, default None
        Random seed of simulation. If None, simulation will be conducted randomly.

    Returns
    -------
    PhenotypeResult
        Dataclass object that includes phenotype and trait dataframe.

    Raises
    ------
    ValueError
        If the number of mutations in `ts` is smaller than `num_causal`.
    ValueError
        If `h2` <= 0 or `h2` > 1

    See Also
    --------
    trait_model : Returns a trait model, which can be used as `model` input.
    PhenotypeResult : Dataclass object that will be used as an output.
    sim_trait : Used to simulate a trait dataframe.
    genetic_value : Used to determine genetic value of individuals.
    sim_env : Used to simulate environmental noise.

    Notes
    -----
    The simulation outputs of traits and phenotypes are given as a
    :py:class:`pandas.DataFrame`.

    The trait dataframe can be extracted by using ``.trait`` in the
    resulting object and contains the following columns:

        * **position**: Position of sites that have causal allele in genome coordinates.
        * **site_id**: Site IDs that have causal allele.
        * **effect_size**: Simulated effect size of causal allele.
        * **causal_allele**: Causal allele.
        * **allele_freq**: Allele frequency of causal allele. It is described in detail
          in :ref:`trait_frequency_dependence`.
        * **trait_id**: Trait ID.

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
        ts=ts,
        model=model,
        num_causal=num_causal,
        causal_sites=causal_sites,
        alpha=alpha,
        random_seed=random_seed,
    )
    genetic_df = tstrait.genetic_value(ts=ts, trait_df=trait_df)
    phenotype_df = tstrait.sim_env(
        genetic_df=genetic_df, h2=h2, random_seed=random_seed
    )

    result = tstrait.PhenotypeResult(trait=trait_df, phenotype=phenotype_df)

    return result


def normalise_phenotypes(phenotype_df, mean=0, var=1, ddof=1):
    """Normalise phenotype dataframe.

    Parameters
    ----------
    phenotype_df : pandas.DataFrame
        Phenotype dataframe.
    mean : float, default 0
        Mean of the resulting phenotype.
    var : float, default 1
        Variance of the resulting phenotype.
    ddof : int, default 1
        Delta degrees of freedom. The divisor used in computing the variance
        is N - ddof, where N represents the number of elements.

    Returns
    -------
    pandas.DataFrame
        Dataframe with normalised phenotype.

    Raises
    ------
    ValueError
        If `var` <= 0.

    Notes
    -----
    The following columns must be included in `phenotype_df`:

        * **trait_id**: Trait ID.
        * **individual_id**: Individual ID.
        * **phenotype**: Simulated phenotypes.

    The dataframe output has the following columns:

        * **trait_id**: Trait ID inside the phenotype_df input.
        * **individual_id**: Individual ID inside the phenotype_df input.
        * **phenotype**: Normalised phenotype.

    Examples
    --------
    See :ref:`normalise_phenotype` section for worked examples.
    """
    if var <= 0:
        raise ValueError("Variance must be greater than 0.")
    phenotype_df = _check_dataframe(
        phenotype_df, ["individual_id", "trait_id", "phenotype"], "phenotype_df"
    )
    grouped = phenotype_df.groupby("trait_id")[["phenotype"]]
    transformed_phenotype = grouped.transform(
        lambda x: (x - x.mean()) / x.std(ddof=ddof)
    )
    transformed_phenotype = transformed_phenotype * np.sqrt(var) + mean
    phenotype_df.loc[:, "phenotype"] = transformed_phenotype

    return phenotype_df
