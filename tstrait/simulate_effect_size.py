import numpy as np
import pandas as pd
import tskit
import tstrait

from .base import _check_instance


class _TraitSimulator:
    """Simulator class to select causal alleles and simulate effect sizes of causal
    mutations.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence data with mutation.
    num_causal : int
        Number of causal sites.
    model : TraitModel
        Trait model that will be used to simulate effect sizes.
    random_seed : int
        The random seed.
    """

    def __init__(self, ts, num_causal, model, random_seed):
        self.ts = ts
        self.num_causal = num_causal
        self.model = model
        self.rng = np.random.default_rng(random_seed)

    def _choose_causal_site(self):
        """Randomly chooses causal site IDs among all the sites in the tree sequence
        data. The site IDs are aligned based on their genomic positions as a part of
        the tree sequence data requirement, so the chosen site IDs are sorted in the
        final step. The algorithm will be faster if the site IDs are aligned by
        their genomic locations.
        """
        site_id = self.rng.choice(
            self.ts.num_sites, size=self.num_causal, replace=False
        )
        site_id = np.sort(site_id)

        return site_id

    def _sim_causal_mutation(self):
        """This method randomly chooses causal sites and the corresponding causal state
        based on the `num_causal` input. Afterwards, effect size of each causal site
        is simulated based on the trait model given by the `model` input.

        Returns
        -------
        pandas.DataFrame
            Trait dataframe that includes causal site ID, causal allele, simulated
            effect size, and trait ID.
        """
        num_trait = self.model.num_trait
        site_id_array = self._choose_causal_site()
        site_id_array = np.repeat(site_id_array, num_trait)

        beta_array = self.model._sim_effect_size(
            num_causal=self.num_causal, rng=self.rng
        )
        trait_id_array = np.tile(np.arange(num_trait), self.num_causal)

        df = pd.DataFrame(
            {
                "site_id": site_id_array,
                "effect_size": beta_array.flatten(),
                "trait_id": trait_id_array,
            }
        )

        return df


def sim_trait(ts, num_causal, model, random_seed=None):
    """
    Randomly selects causal sites and simulates effect sizes.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence data that will be used in the quantitative trait
        simulation.
    num_causal : int
        Number of causal sites.
    model : tstrait.TraitModel
        Trait model that will be used to simulate effect sizes.
    random_seed : int, default None
        Random seed of simulation. If None, simulation will be conducted randomly.

    Returns
    -------
    pandas.DataFrame
        Trait dataframe that includes simulated effect sizes.

    Raises
    ------
    ValueError
        If the number of mutations in `ts` is smaller than `num_causal`.

    See Also
    --------
    trait_model : Return a trait model, which can be used as `model` input.
    sim_genetic : The trait dataframe output can be used as an input to simulate
        genetic values.

    Notes
    -----
    The simulation output is given as a :py:class:`pandas.DataFrame` and contains the
    following columns:

        * **site_id**: Site IDs that have causal mutation.
        * **effect_size**: Simulated effect size of causal mutation.
        * **trait_id**: Trait ID.

    Examples
    --------
    See :ref:`effect_size_sim` for worked examples.
    """
    ts = _check_instance(ts, "ts", tskit.TreeSequence)
    model = _check_instance(model, "model", tstrait.TraitModel)
    num_sites = ts.num_sites
    if num_sites == 0:
        raise ValueError("No mutation in the tree sequence input")
    if num_causal > num_sites:
        raise ValueError(
            "num_causal must be an integer not greater than the number of sites in ts"
        )

    simulator = _TraitSimulator(
        ts=ts,
        num_causal=num_causal,
        model=model,
        random_seed=random_seed,
    )
    trait_df = simulator._sim_causal_mutation()

    return trait_df
