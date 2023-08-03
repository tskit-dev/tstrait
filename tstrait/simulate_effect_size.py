import collections

import numpy as np
import pandas as pd
import tskit
import tstrait

from .base import _define_rng, _check_val, _check_int, _check_instance  # noreorder


class TraitSimulator:
    """Simulator class to select causal alleles and simulate effect sizes of causal
    mutations.

    :param ts: Tree sequence data with mutation.
    :type ts: tskit.TreeSequence
    :param num_causal: Number of causal sites.
    :type num_causal: int
    :param model: Trait model that will be used to simulate effect sizes.
    :type model: TraitModel
    :param alpha: Parameter that determines the emphasis placed on rarer variants.
    :type alpha: float
    :param random_seed: The random seed. If this is not specified or None, simulation
        will be done randomly.
    :type random_seed: None or int
    """

    def __init__(self, ts, num_causal, model, alpha, random_seed):
        self.ts = ts
        self.num_causal = num_causal
        self.model = model
        self.alpha = alpha
        self.rng = _define_rng(random_seed)

    def _choose_causal_site(self):
        """Randomly chooses causal site IDs among all the sites in the tree sequence
        data. The site IDs are aligned based on their genomic positions as a part of
        the tree sequence data requirement, so the chosen site IDs are sorted in the
        final step. The algorithm will be faster if the site IDs are aligned by
        their genomic locations.
        """
        site_id = self.rng.choice(
            range(self.ts.num_sites), size=self.num_causal, replace=False
        )
        site_id = np.sort(site_id)

        return site_id

    def _obtain_allele_count(self, tree, site):
        """Obtain a dictionary of allele counts, and the ancestral state is not
        included in the dictionary. Input is the tree sequence site (`ts.site(ID)`)
        instead of site ID, as obtaining `ts.site(ID)` can be time consuming. The
        ancestral state is not deleted if the ancestral state is the only allele
        at that site.
        """
        counts = collections.Counter({site.ancestral_state: self.ts.num_samples})
        for m in site.mutations:
            current_state = site.ancestral_state
            if m.parent != tskit.NULL:
                current_state = self.ts.mutation(m.parent).derived_state
            # Silent mutations do nothing
            if current_state != m.derived_state:
                num_samples = tree.num_samples(m.node)
                counts[m.derived_state] += num_samples
                counts[current_state] -= num_samples
        del counts[site.ancestral_state]
        counts = {x: y for x, y in counts.items() if y != 0}
        if len(counts) == 0:
            counts = {site.ancestral_state: self.ts.num_samples}
        return counts

    def _frequency_dependence(self, allele_freq):
        if allele_freq == 0 or allele_freq == 1:
            const = 0
        else:
            const = np.sqrt(pow(2 * allele_freq * (1 - allele_freq), self.alpha))
        return const

    def sim_causal_mutation(self):
        """This method randomly chooses causal sites and the corresponding causal state
        based on the `num_causal` input. Afterwards, effect size of each causal site
        is simulated based on the trait model given by the `model` input.

        :return: Returns a pandas dataframe that includes causal site ID, causal allele,
            simulated effect size, and trait ID.
        :rtype: pandas.DataFrame
        """
        causal_site_array = self._choose_causal_site()
        num_samples = self.ts.num_samples
        tree = tskit.Tree(self.ts)

        causal_state_array = np.zeros(self.num_causal, dtype=object)
        beta_array = np.zeros((self.num_causal, self.model.num_trait))

        for i, single_id in enumerate(causal_site_array):
            site = self.ts.site(single_id)
            tree.seek(site.position)
            counts = self._obtain_allele_count(tree, site)
            causal_state = self.rng.choice(list(counts))
            causal_state_array[i] = causal_state
            allele_freq = counts[causal_state] / num_samples
            beta = self.model.sim_effect_size(num_causal=self.num_causal, rng=self.rng)
            beta *= self._frequency_dependence(allele_freq)
            beta_array[i] = beta

        df = pd.DataFrame(
            columns=["site_id", "causal_state", "effect_size", "trait_id"]
        )

        for i in range(self.model.num_trait):
            df_add = pd.DataFrame(
                {
                    "site_id": causal_site_array,
                    "causal_state": causal_state_array,
                    "effect_size": beta_array[:, i],
                    "trait_id": np.ones(self.num_causal) * i,
                }
            )
            df = pd.concat([df, df_add])

        df = df.reset_index()
        del df["index"]

        return df


def sim_trait(ts, num_causal, model, alpha=0, random_seed=None):
    """Randomly selects causal sites from the inputted tree sequence data, and simulates
    effect sizes of causal mutations.

    :param ts: The tree sequence data that will be used in the quantitative trait
        simulation. The tree sequence data must include a mutation.
    :type ts: tskit.TreeSequence
    :param num_causal: Number of causal sites that will be chosen randomly. It must be
        a positive integer that is less than the number of sites in the tree sequence
        data.
    :type num_causal: int
    :param model: Trait model that will be used to simulate effect sizes of causal
        mutations.
    :type model: TraitModel
    :param alpha: Parameter that determines the relative weight on rarer variants.
        A negative `alpha` value can increase the magnitude of effect sizes coming
        from rarer variants. The frequency dependent architecture can be ignored
        by setting `alpha` to be zero.
    :type alpha: float
    :param random_seed: The random seed. If this is not specified or None, simulation
        will be done randomly.
    :type random_seed: None or int
    :return: Returns a pandas dataframe that includes causal site ID, causal allele and
        simulated effect size. It can be used as an input in :func:`sim_phenotype`
        function to simulate phenotypes.
    :rtype: pandas.DataFrame
    """
    ts = _check_instance(ts, "ts", tskit.TreeSequence)
    num_causal = _check_int(num_causal, "num_causal", minimum=0)
    model = _check_instance(model, "model", tstrait.TraitModel)
    alpha = _check_val(alpha, "alpha")
    num_sites = ts.num_sites
    if num_sites == 0:
        raise ValueError("No mutation in the tree sequence input")
    if num_causal > num_sites:
        raise ValueError(
            "num_causal must be an integer not greater than the number of sites in ts"
        )

    simulator = TraitSimulator(
        ts=ts,
        num_causal=num_causal,
        model=model,
        alpha=alpha,
        random_seed=random_seed,
    )
    effect_size_df = simulator.sim_causal_mutation()

    return effect_size_df
