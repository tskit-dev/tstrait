from dataclasses import dataclass

import numpy as np
import pandas as pd
import tskit
import tstrait

from .base import _check_instance
from .base import _check_int
from .base import _check_val


@dataclass
class _FreqResult:
    """
    Data class that contains simulated effect size and allele frequency.

    Attributes
    ----------
    beta_array : numpy.array
        Numpy array that includes simulated effect size.
    allele_freq : numpy.array
        Allele frequency of each causal mutation.
    """

    beta_array: np.array
    allele_freq: np.array


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
    alpha : float
        Parameter that determines the degree of the frequency dependence model.
    random_seed : int
        The random seed.
    """

    def __init__(self, ts, num_causal, model, alpha, random_seed):
        self.ts = ts
        self.num_causal = num_causal
        self.model = model
        self.alpha = alpha
        self.rng = np.random.default_rng(random_seed)

    def _choose_causal_site(self):
        """
        Randomly chooses causal site IDs among all the sites in the tree sequence
        data. The site IDs are aligned based on their genomic positions as a part of
        the tree sequence data requirement, so the chosen site IDs are sorted in the
        final step. The algorithm will be faster if the site IDs are aligned by
        their genomic locations.
        """
        site_id = self.rng.choice(
            self.ts.num_sites, size=self.num_causal, replace=False
        )
        return np.sort(site_id)

    def _choose_causal_allele(self, site_id):
        """
        Randomly chooses a causal allele by choosing a causal mutation at random.
        """
        causal_allele = self.rng.choice(self.ts.site(site_id).mutations).derived_state

        return causal_allele

    def _obtain_allele_count(self, tree, site, causal_allele):
        """
        Obtain number of samples with the `causal_allele` in a tree. Input is the tree
        sequence site (`ts.site(ID)`) instead of site ID, as obtaining `ts.site(ID)` can
        be time consuming.
        """
        if site.ancestral_state == causal_allele:
            counts = self.ts.num_samples
        else:
            counts = 0
        for m in site.mutations:
            current_state = site.ancestral_state
            if m.parent != tskit.NULL:
                current_state = self.ts.mutation(m.parent).derived_state
            # Silent mutations do nothing
            if current_state != m.derived_state:
                if m.derived_state == causal_allele:
                    counts += tree.num_samples(m.node)
                elif current_state == causal_allele:
                    counts -= tree.num_samples(m.node)

        return counts

    def _frequency_multiplier(self, allele_freq):
        """
        Calculates the frequency dependence constant [2p(1-p)]^alpha by using
        the allele_freq input.
        """
        if allele_freq == 0 or allele_freq == 1:
            const = 0
        else:
            const = np.sqrt(pow(2 * allele_freq * (1 - allele_freq), self.alpha))
        return const

    def _freq_dep(self, site_id_array, causal_allele_array):
        """
        Obtains the frequency dependent constant [2p(1-p)]^alpha for each site ID.
        """
        tree = tskit.Tree(self.ts)
        num_samples = self.ts.num_samples
        freq_dep_array = []
        allele_freq_array = []
        for site_id, causal_allele in zip(site_id_array, causal_allele_array):
            site = self.ts.site(site_id)
            tree.seek(site.position)
            freq = self._obtain_allele_count(tree, site, causal_allele) / num_samples
            freq_dep_array.append(self._frequency_multiplier(freq))
            allele_freq_array.append(freq)

        result = _FreqResult(beta_array=freq_dep_array, allele_freq=allele_freq_array)

        return result

    def _sim_beta(self, site_id_array, causal_allele_array):
        """
        Simulates effect size by using the model given in the `model` input. If `alpha`
        is non-zero, frequency dependence architecture is used.
        """
        beta_array_non_freq = self.model._sim_effect_size(
            num_causal=self.num_causal, rng=self.rng
        )

        result = self._freq_dep(site_id_array, causal_allele_array)
        result.beta_array = np.multiply(
            result.beta_array, np.transpose(beta_array_non_freq)
        )

        return result

    def _run(self):
        """
        This method runs a simulation based on the input parameters. This method randomly
        chooses causal sites and the corresponding causal allele based on the
        `num_causal` input. Afterwards, effect size of each causal site is simulated
        based on the trait model given by the `model` input. If `alpha` is non-zero,
        frequency dependence architecture is used.
        """
        site_id_array = self._choose_causal_site()
        causal_allele_array = []
        for site_id in site_id_array:
            causal_allele_array.append(self._choose_causal_allele(site_id))

        result = self._sim_beta(site_id_array, causal_allele_array)

        num_trait = self.model.num_trait

        position = self.ts.sites_position[site_id_array]
        if np.array_equal(np.floor(position), position):  # pragma: no cover
            position = position.astype(int)

        trait_df = pd.DataFrame(
            {
                "position": np.repeat(position, num_trait),
                "site_id": np.repeat(site_id_array, num_trait),
                "effect_size": result.beta_array.flatten(order="F"),
                "causal_allele": np.repeat(causal_allele_array, num_trait),
                "allele_freq": np.repeat(result.allele_freq, num_trait),
                "trait_id": np.tile(np.arange(num_trait), self.num_causal),
            }
        )

        return trait_df


def sim_trait(ts, model, *, num_causal=None, alpha=None, random_seed=None):
    """
    Randomly selects causal sites and the corresponding causal allele, and simulates
    effect sizes for each of the chosen causal site.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence data that will be used in the quantitative trait
        simulation.
    model : tstrait.TraitModel
        Trait model that will be used to simulate effect sizes.
    num_causal : int, default None
        Number of causal sites. If None, number of causal sites will be 1.
    alpha : float, default None
        Parameter that determines the degree of the frequency dependence model. Please
        see :ref:`frequency_dependence` for details on how this parameter influences
        effect size simulation. If None, alpha will be 0.
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
    genetic_value : The trait dataframe output can be used as an input to obtain
        genetic values.

    Notes
    -----
    The simulation output is given as a :py:class:`pandas.DataFrame` and contains the
    following columns:

        * **position**: Position of sites that have causal allele in genome coordinates.
        * **site_id**: Site IDs that have causal allele.
        * **effect_size**: Simulated effect size of causal allele.
        * **causal_allele**: Causal allele.
        * **allele_freq**: Allele frequency of causal allele. It is described in detail
          in :ref:`trait_frequency_dependence`.
        * **trait_id**: Trait ID.

    Examples
    --------
    See :ref:`sim_trait` for worked examples.
    """
    ts = _check_instance(ts, "ts", tskit.TreeSequence)
    model = _check_instance(model, "model", tstrait.TraitModel)
    num_causal = 1 if num_causal is None else num_causal
    num_causal = _check_int(num_causal, "num_causal", minimum=1)
    alpha = 0 if alpha is None else alpha
    alpha = _check_val(alpha, "alpha")
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
        alpha=alpha,
        random_seed=random_seed,
    )
    trait_df = simulator._run()

    return trait_df
