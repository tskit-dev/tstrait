import collections
import numbers

import numpy as np
import pandas as pd
import tskit
import tstrait


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
    :param site_id: Causal site ID. If this is None, causal site IDs will be randomly
        chosen.
    :type site_id: None or np.ndarray
    :param random_seed: The random seed. If this is not specified or None, simulation
        will be done randomly.
    :type random_seed: None or int
    """

    def __init__(self, ts, num_causal, model, alpha, site_id, random_seed):
        self.ts = ts
        self.num_causal = num_causal
        self.model = model
        self.alpha = alpha
        self.site_id = site_id
        self.rng = np.random.default_rng(random_seed)

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

        :return: Returns a pandas dataframe that includes causal site ID, causal allele
            and simulated effect size
        :rtype: pandas.DataFrame
        """
        if self.site_id is None:
            self.site_id = self._choose_causal_site()

        num_samples = self.ts.num_samples
        tree = tskit.Tree(self.ts)

        causal_state_array = np.zeros(self.num_causal, dtype=object)
        beta_array = np.zeros(self.num_causal)

        for i, single_id in enumerate(self.site_id):
            site = self.ts.site(single_id)
            tree.seek(site.position)
            counts = self._obtain_allele_count(tree, site)
            causal_state = self.rng.choice(list(counts))
            causal_state_array[i] = causal_state
            allele_freq = counts[causal_state] / num_samples
            beta = self.model.sim_effect_size(num_causal=self.num_causal, rng=self.rng)
            beta *= self._frequency_dependence(allele_freq)
            beta_array[i] = beta

        effect_size_df = pd.DataFrame(
            {
                "site_id": self.site_id,
                "causal_state": causal_state_array,
                "effect_size": beta_array,
            }
        )

        return effect_size_df


def sim_trait(ts, causal_sites, model, alpha=0, random_seed=None):
    """Randomly selects causal sites from the inputted tree sequence data, and simulates
    effect sizes of causal mutations.

    :param ts: The tree sequence data that will be used in the quantitative trait
        simulation. The tree sequence data must include a mutation.
    :type ts: tskit.TreeSequence
    :param causal_sites: The parameter that determines the causal site IDs. When it is
        a positive integer, it is read as the number of causal sites, and causal site IDs
        will be chosen randomly. When it is a list or numpy array, it will be used as
        causal site IDs.
    :type causal_sites: int or list or numpy.ndarray
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
    if not isinstance(ts, tskit.TreeSequence):
        raise TypeError("Input must be a tree sequence data")
    if (
        not isinstance(causal_sites, numbers.Number)
        and not isinstance(causal_sites, list)
        and not isinstance(causal_sites, np.ndarray)
    ):
        raise TypeError("causal_sites must be an integer or a list or a numpy array")

    if isinstance(causal_sites, numbers.Number):
        num_causal = causal_sites
        if int(causal_sites) != causal_sites or causal_sites <= 0:
            raise ValueError("Number of causal sites must be a positive integer")
        causal_sites = None
    else:
        causal_sites = np.array(causal_sites)
        causal_sites = np.unique(causal_sites)
        num_causal = len(causal_sites)

    if not isinstance(model, tstrait.TraitModel):
        raise TypeError("Trait model must be an instance of TraitModel")
    num_sites = ts.num_sites
    if num_sites == 0:
        raise ValueError("No mutation in the provided data")
    if num_causal > num_sites:
        raise ValueError(
            "There are less number of sites in the tree sequence than the inputted "
            "number of causal sites"
        )
    if not isinstance(alpha, numbers.Number):
        raise TypeError("Alpha must be a number")

    simulator = TraitSimulator(
        ts=ts,
        num_causal=num_causal,
        model=model,
        alpha=alpha,
        site_id=causal_sites,
        random_seed=random_seed,
    )
    effect_size_df = simulator.sim_causal_mutation()

    return effect_size_df
