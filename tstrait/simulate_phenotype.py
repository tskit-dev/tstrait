from dataclasses import dataclass

import pandas as pd
import tstrait


@dataclass
class Result:
    """Data class that contains effect size dataframe and phenotype dataframe.

    :param effect_size: Effect size dataframe that includes site_id, causal_state,
        effect_size, and trait_id
    :param phenotype: Phenotype dataframe that includes trait_id, individual_id,
        genetic_value, environmental_noise, and phenotype
    :type phenotype: pandas.DataFrame
    """

    effect_size: pd.DataFrame
    phenotype: pd.DataFrame


def sim_phenotype(ts, num_causal, model, h2, alpha=0, random_seed=None):
    """Simulates quantitative traits of individuals based on the inputted tree sequence
    and the specified trait model, and returns a :class:`Result` object.

    :param ts: The tree sequence data that will be used in the quantitative trait
        simulation. The tree sequence data must include a mutation.
    :type ts: tskit.TreeSequence
    :param num_causal: Number of causal sites that will be chosen randomly. It should
        be a positive integer that is greater than the number of sites in the tree
        sequence data.
    :type num_causal: int
    :param model: Trait model that will be used to simulate effect sizes of causal sites.
    :type model: tstrait.TraitModel
    :param h2: Narrow-sense heritability, which will be used to simulate environmental
        noise. Narrow-sense heritability must be between 0 and 1.
    :type h2: float
    :param alpha: Parameter that determines the relative weight on rarer variants.
        A negative `alpha` value can increase the magnitude of effect sizes coming
        from rarer variants. The frequency dependent architecture can be ignored
        by setting `alpha` to be zero.
    :type alpha: float
    :param random_seed: The random seed. If this is not specified or None, simulation
        will be done randomly.
    :type random_seed: None or int
    :return: Returns the :class:`Result` object that includes the effect size
        dataframe and phenotype dataframe.
    :rtype: Result
    """
    trait_df = tstrait.sim_trait(
        ts=ts, num_causal=num_causal, model=model, alpha=alpha, random_seed=random_seed
    )
    genetic_df = tstrait.genetic_value(ts=ts, trait_df=trait_df)
    phenotype_df = tstrait.sim_env(
        genetic_df=genetic_df, h2=h2, random_seed=random_seed
    )

    result = tstrait.Result(effect_size=trait_df, phenotype=phenotype_df)

    return result
