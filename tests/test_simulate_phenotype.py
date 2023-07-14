import functools

import msprime
import numpy as np
import pandas as pd
import pytest
import tskit
import tstrait


@functools.lru_cache(maxsize=None)
def all_trees_ts(n):
    """
    Generate a tree sequence that corresponds to the lexicographic listing
    of all trees with n leaves (i.e. from tskit.all_trees(n)).

    Note: it would be nice to include a version of this in the combinatorics
    module at some point but the implementation is quite inefficient. Also
    it's not entirely clear that the way we're allocating node times is
    guaranteed to work.
    """
    tables = tskit.TableCollection(0)
    for _ in range(n):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for j in range(1, n):
        tables.nodes.add_row(flags=0, time=j)

    L = 0
    for tree in tskit.all_trees(n):
        for u in tree.preorder()[1:]:
            tables.edges.add_row(L, L + 1, tree.parent(u), u)
        L += 1
    tables.sequence_length = L
    tables.sort()
    tables.simplify()
    return tables.tree_sequence()


class Test_input:
    def sample_effect_size_df(self):
        data = [[1, "A", 0.1]]
        df = pd.DataFrame(data, columns=["SiteID", "CausalState", "EffectSize"])
        return df

    @pytest.mark.parametrize("ts", [10, "a", True])
    def test_input_tree_sequence(self, ts):
        effect_size_df = self.sample_effect_size_df()
        with pytest.raises(TypeError, match="Input must be a tree sequence data"):
            tstrait.sim_phenotype(
                ts=ts, effect_size=effect_size_df, h2=0.3, random_seed=1
            )

    @pytest.mark.parametrize("effect_size_df", [{"A": 1}, 1, [1, 1]])
    def test_input_effect_size_df(self, effect_size_df):
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        with pytest.raises(
            TypeError, match="Effect size input must be a pandas dataframe"
        ):
            tstrait.sim_phenotype(
                ts=ts, effect_size=effect_size_df, h2=0.3, random_seed=1
            )

    def test_input_col1(self):
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        data = [[1, "A", 0.1]]
        effect_size_df = pd.DataFrame(
            data, columns=["Site", "CausalState", "EffectSize"]
        )
        with pytest.raises(
            ValueError, match="SiteID is not included in the effect size dataframe"
        ):
            tstrait.sim_phenotype(
                ts=ts, effect_size=effect_size_df, h2=0.3, random_seed=1
            )

    def test_input_col2(self):
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        data = [[1, "A", 0.1]]
        effect_size_df = pd.DataFrame(data, columns=["SiteID", "State", "EffectSize"])
        with pytest.raises(
            ValueError, match="CausalState is not included in the effect size dataframe"
        ):
            tstrait.sim_phenotype(
                ts=ts, effect_size=effect_size_df, h2=0.3, random_seed=1
            )

    def test_input_col3(self):
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        data = [[1, "A"]]
        effect_size_df = pd.DataFrame(data, columns=["SiteID", "CausalState"])
        with pytest.raises(
            ValueError, match="EffectSize is not included in the effect size dataframe"
        ):
            tstrait.sim_phenotype(
                ts=ts, effect_size=effect_size_df, h2=0.3, random_seed=1
            )


class Test_sim_phenotype_output_dim:
    @pytest.mark.parametrize("num_ind", [10, np.array([5])[0]])
    @pytest.mark.parametrize("num_causal", [1, 2, np.array([3])[0]])
    @pytest.mark.parametrize("alpha", [0, 1, -1.1])
    @pytest.mark.parametrize("random_seed", [1, 2, None])
    def test_output_dim_normal(self, num_ind, num_causal, alpha, random_seed):
        h2 = 0.3
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        effect_size_df = tstrait.sim_traits(
            ts=ts,
            num_causal=num_causal,
            model=model,
            alpha=alpha,
            random_seed=random_seed,
        )
        sim_result = tstrait.sim_phenotype(
            ts=ts, effect_size=effect_size_df, h2=h2, random_seed=random_seed
        )
        assert sim_result.shape[0] == num_ind
        assert sim_result.shape[1] == 4

        assert np.array_equal(sim_result["IndividualID"], np.arange(num_ind))
        assert np.array_equal(
            sim_result["Phenotype"],
            sim_result["GeneticValue"] + sim_result["EnvironmentalNoise"],
        )


class Test_heritability:
    @pytest.mark.parametrize("num_ind", [1, 2, 5])
    @pytest.mark.parametrize("num_causal", [1, 2, 3])
    @pytest.mark.parametrize("random_seed", [1, 2])
    def test_h2_one(self, num_ind, num_causal, random_seed):
        h2 = 1
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        effect_size_df = tstrait.sim_traits(
            ts=ts, num_causal=num_causal, model=model, alpha=1, random_seed=random_seed
        )
        sim_result = tstrait.sim_phenotype(
            ts=ts, effect_size=effect_size_df, h2=h2, random_seed=random_seed
        )

        assert np.allclose(sim_result["Phenotype"], sim_result["GeneticValue"])
        assert np.array_equal(sim_result["EnvironmentalNoise"], np.zeros(num_ind))

    @pytest.mark.parametrize("h2", [0, -1, 2])
    def test_h2_error(self, h2):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        effect_size_df = tstrait.sim_traits(
            ts=ts, num_causal=3, model=model, alpha=1, random_seed=1
        )
        with pytest.raises(ValueError, match="Heritability must be 0 < h2 <= 1"):
            tstrait.sim_phenotype(
                ts=ts, effect_size=effect_size_df, h2=h2, random_seed=1
            )

    @pytest.mark.parametrize("h2", ["A", "0.1", [0.1, 0.3]])
    def test_h2_error_input(self, h2):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        effect_size_df = tstrait.sim_traits(
            ts=ts, num_causal=3, model=model, alpha=1, random_seed=1
        )
        with pytest.raises(TypeError, match="Heritability must be a number"):
            tstrait.sim_phenotype(
                ts=ts, effect_size=effect_size_df, h2=h2, random_seed=1
            )

    @pytest.mark.parametrize("h2", [0.3, 1])
    def test_ind_one(self, h2):
        ts = msprime.sim_ancestry(1, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        effect_size_df = tstrait.sim_traits(
            ts=ts, num_causal=1, model=model, alpha=1, random_seed=1
        )
        sim_result = tstrait.sim_phenotype(
            ts=ts, effect_size=effect_size_df, h2=h2, random_seed=1
        )

        assert np.array_equal(sim_result["EnvironmentalNoise"], np.zeros(1))
        assert np.array_equal(sim_result["Phenotype"], sim_result["GeneticValue"])


class Test_site_genotypes:
    def binary_tree(self):
        #  3.00   6
        #     ┊ ┏━┻━┓    ┊
        # 2.00┊ ┃   5   ┊
        #     ┊ ┃ ┏━┻┓   ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓  ┊
        # 0.00 0 1 2 3
        #
        ts = tskit.Tree.generate_comb(4, span=15).tree_sequence
        tables = ts.dump_tables()
        for j in range(12):
            tables.sites.add_row(j, "A")

        tables.individuals.add_row()
        tables.individuals.add_row()
        individuals = tables.nodes.individual
        individuals[0] = 0
        individuals[1] = 0
        individuals[2] = 1
        individuals[3] = 1
        tables.nodes.individual = individuals

        tables.mutations.add_row(site=0, node=0, derived_state="T")
        tables.mutations.add_row(site=1, node=4, derived_state="T")
        tables.mutations.add_row(site=2, node=1, derived_state="T")

        tables.mutations.add_row(site=3, node=5, derived_state="T")
        tables.mutations.add_row(site=3, node=2, derived_state="T", parent=3)

        tables.mutations.add_row(site=4, node=0, derived_state="T")
        tables.mutations.add_row(site=4, node=4, derived_state="T")

        tables.mutations.add_row(site=5, node=0, derived_state="T")
        tables.mutations.add_row(site=5, node=0, derived_state="A", parent=7)

        tables.mutations.add_row(site=6, node=0, derived_state="T")
        tables.mutations.add_row(site=6, node=0, derived_state="G", parent=9)
        tables.mutations.add_row(site=6, node=0, derived_state="T", parent=10)
        tables.mutations.add_row(site=6, node=0, derived_state="C", parent=11)

        tables.mutations.add_row(site=7, node=5, derived_state="T")
        tables.mutations.add_row(site=7, node=4, derived_state="C", parent=13)

        tables.mutations.add_row(site=8, node=5, derived_state="T")
        tables.mutations.add_row(site=8, node=4, derived_state="C", parent=15)
        tables.mutations.add_row(site=8, node=4, derived_state="T", parent=16)
        tables.mutations.add_row(site=8, node=4, derived_state="A", parent=17)
        tables.mutations.add_row(site=8, node=4, derived_state="T", parent=18)

        tables.mutations.add_row(site=9, node=6, derived_state="A")

        tables.mutations.add_row(site=10, node=6, derived_state="C")

        tables.mutations.add_row(site=11, node=5, derived_state="C")
        tables.mutations.add_row(site=11, node=0, derived_state="T")

        ts = tables.tree_sequence()

        return ts

    def binary_tree_df(self):
        effect_size = {
            "SiteID": np.arange(12),
            "CausalState": ["T", "T", "T", "T", "T", "A", "C", "T", "T", "A", "C", "C"],
            "EffectSize": np.ones(12),
        }
        effect_size_df = pd.DataFrame(effect_size)

        return effect_size_df

    def test_binary_tree_genotype(self):
        ts = self.binary_tree()
        effect_size_df = self.binary_tree_df()
        simulator = tstrait.PhenotypeSimulator(
            ts=ts, effect_size_df=effect_size_df, h2=0.1, random_seed=1
        )
        tree = ts.first()

        g1 = simulator._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g2 = simulator._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g3 = simulator._individual_genotype(tree, ts.site(2), "T", ts.num_nodes)
        g4 = simulator._individual_genotype(tree, ts.site(3), "T", ts.num_nodes)
        g5 = simulator._individual_genotype(tree, ts.site(4), "T", ts.num_nodes)
        g6 = simulator._individual_genotype(tree, ts.site(5), "A", ts.num_nodes)
        g7 = simulator._individual_genotype(tree, ts.site(6), "C", ts.num_nodes)
        g8 = simulator._individual_genotype(tree, ts.site(7), "T", ts.num_nodes)
        g9 = simulator._individual_genotype(tree, ts.site(8), "T", ts.num_nodes)
        g10 = simulator._individual_genotype(tree, ts.site(9), "A", ts.num_nodes)
        g11 = simulator._individual_genotype(tree, ts.site(10), "C", ts.num_nodes)
        g12 = simulator._individual_genotype(tree, ts.site(11), "C", ts.num_nodes)

        assert np.array_equal(g1, np.array([1, 0]))
        assert np.array_equal(g2, np.array([0, 2]))
        assert np.array_equal(g3, np.array([1, 0]))
        assert np.array_equal(g4, np.array([1, 2]))
        assert np.array_equal(g5, np.array([1, 2]))
        assert np.array_equal(g6, np.array([2, 2]))
        assert np.array_equal(g7, np.array([1, 0]))
        assert np.array_equal(g8, np.array([1, 0]))
        assert np.array_equal(g9, np.array([1, 2]))
        assert np.array_equal(g10, np.array([2, 2]))
        assert np.array_equal(g11, np.array([2, 2]))
        assert np.array_equal(g12, np.array([1, 2]))

    def test_binary_tree_output(self):
        ts = self.binary_tree()
        effect_size_df = self.binary_tree_df()
        sim_result = tstrait.sim_phenotype(
            ts=ts, effect_size=effect_size_df, h2=0.1, random_seed=1
        )
        assert np.array_equal(sim_result["GeneticValue"], np.array([14, 16]))

    def binary_tree_different_individual(self):
        #  3.00   6
        #     ┊ ┏━┻━┓    ┊
        # 2.00┊ ┃   5   ┊
        #     ┊ ┃ ┏━┻┓   ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓  ┊
        # 0.00 0 1 2 3
        #
        ts = tskit.Tree.generate_comb(4, span=10).tree_sequence
        tables = ts.dump_tables()
        for j in range(9):
            tables.sites.add_row(j, "A")

        tables.individuals.add_row()
        tables.individuals.add_row()
        individuals = tables.nodes.individual
        individuals[0] = 0
        individuals[1] = 1
        individuals[2] = 0
        individuals[3] = 1
        tables.nodes.individual = individuals

        tables.mutations.add_row(site=0, node=0, derived_state="T")
        tables.mutations.add_row(site=1, node=4, derived_state="T")
        tables.mutations.add_row(site=2, node=1, derived_state="T")

        tables.mutations.add_row(site=3, node=5, derived_state="T")
        tables.mutations.add_row(site=3, node=2, derived_state="T", parent=3)

        tables.mutations.add_row(site=4, node=0, derived_state="T")
        tables.mutations.add_row(site=4, node=4, derived_state="T")

        tables.mutations.add_row(site=5, node=0, derived_state="T")
        tables.mutations.add_row(site=5, node=0, derived_state="A", parent=7)

        tables.mutations.add_row(site=6, node=0, derived_state="T")
        tables.mutations.add_row(site=6, node=0, derived_state="G", parent=9)
        tables.mutations.add_row(site=6, node=0, derived_state="T", parent=10)
        tables.mutations.add_row(site=6, node=0, derived_state="C", parent=11)

        tables.mutations.add_row(site=7, node=5, derived_state="T")
        tables.mutations.add_row(site=7, node=4, derived_state="C", parent=13)

        tables.mutations.add_row(site=8, node=5, derived_state="T")
        tables.mutations.add_row(site=8, node=4, derived_state="C", parent=15)
        tables.mutations.add_row(site=8, node=4, derived_state="T", parent=16)
        tables.mutations.add_row(site=8, node=4, derived_state="A", parent=17)
        tables.mutations.add_row(site=8, node=4, derived_state="T", parent=18)
        ts = tables.tree_sequence()

        return ts

    def binary_tree_different_individual_df(self):
        effect_size = {
            "SiteID": np.arange(9),
            "CausalState": ["T", "T", "T", "T", "T", "A", "C", "T", "T"],
            "EffectSize": np.ones(9),
        }
        effect_size_df = pd.DataFrame(effect_size)
        return effect_size_df

    def test_binary_tree_different_individual_genotype(self):
        ts = self.binary_tree_different_individual()
        effect_size_df = self.binary_tree_different_individual_df()
        simulator = tstrait.PhenotypeSimulator(
            ts=ts, effect_size_df=effect_size_df, h2=0.1, random_seed=1
        )
        tree = ts.first()

        g1 = simulator._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g2 = simulator._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g3 = simulator._individual_genotype(tree, ts.site(2), "T", ts.num_nodes)
        g4 = simulator._individual_genotype(tree, ts.site(3), "T", ts.num_nodes)
        g5 = simulator._individual_genotype(tree, ts.site(4), "T", ts.num_nodes)
        g6 = simulator._individual_genotype(tree, ts.site(5), "A", ts.num_nodes)
        g7 = simulator._individual_genotype(tree, ts.site(6), "C", ts.num_nodes)
        g8 = simulator._individual_genotype(tree, ts.site(7), "T", ts.num_nodes)
        g9 = simulator._individual_genotype(tree, ts.site(8), "T", ts.num_nodes)

        assert np.array_equal(g1, np.array([1, 0]))
        assert np.array_equal(g2, np.array([1, 1]))
        assert np.array_equal(g3, np.array([0, 1]))
        assert np.array_equal(g4, np.array([1, 2]))
        assert np.array_equal(g5, np.array([2, 1]))
        assert np.array_equal(g6, np.array([2, 2]))
        assert np.array_equal(g7, np.array([1, 0]))
        assert np.array_equal(g8, np.array([0, 1]))
        assert np.array_equal(g9, np.array([1, 2]))

    def test_binary_tree_different_individual_output(self):
        ts = self.binary_tree_different_individual()
        effect_size_df = self.binary_tree_different_individual_df()
        sim_result = tstrait.sim_phenotype(
            ts=ts, effect_size=effect_size_df, h2=0.1, random_seed=1
        )
        assert np.array_equal(sim_result["GeneticValue"], np.array([9, 10]))

    def tree_internal_node(self):
        ts = tskit.Tree.generate_balanced(4, span=10).tree_sequence

        tables = ts.dump_tables()
        for j in range(3):
            tables.sites.add_row(j, "A")

        tables.individuals.add_row()
        tables.individuals.add_row()
        tables.individuals.add_row()
        individuals = tables.nodes.individual
        individuals[0] = 1
        individuals[1] = 1
        individuals[2] = 2
        individuals[3] = 2
        individuals[4] = 0
        individuals[5] = 0
        tables.nodes.individual = individuals

        tables.mutations.add_row(site=0, node=0, derived_state="T")
        tables.mutations.add_row(site=0, node=5, derived_state="G")

        tables.mutations.add_row(site=1, node=4, derived_state="T")
        tables.mutations.add_row(site=1, node=0, derived_state="A", parent=2)
        tables.mutations.add_row(site=1, node=5, derived_state="T")
        tables.mutations.add_row(site=1, node=5, derived_state="G", parent=4)

        tables.mutations.add_row(site=2, node=4, derived_state="T")
        tables.mutations.add_row(site=2, node=0, derived_state="G", parent=6)
        tables.mutations.add_row(site=2, node=1, derived_state="C", parent=6)

        ts = tables.tree_sequence()

        return ts

    def tree_internal_node_df(self):
        effect_size = {
            "SiteID": np.arange(3),
            "CausalState": ["G", "T", "T"],
            "EffectSize": np.ones(3),
        }
        effect_size_df = pd.DataFrame(effect_size)
        return effect_size_df

    def test_binary_tree_internal_node_genotype(self):
        ts = self.tree_internal_node()
        effect_size_df = self.tree_internal_node_df()
        simulator = tstrait.PhenotypeSimulator(
            ts=ts, effect_size_df=effect_size_df, h2=0.1, random_seed=1
        )
        tree = ts.first()
        g1 = simulator._individual_genotype(tree, ts.site(0), "G", ts.num_nodes)
        g2 = simulator._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g3 = simulator._individual_genotype(tree, ts.site(2), "T", ts.num_nodes)
        g4 = simulator._individual_genotype(tree, ts.site(2), "G", ts.num_nodes)

        assert np.array_equal(g1, np.array([1, 0, 2]))
        assert np.array_equal(g2, np.array([1, 1, 0]))
        assert np.array_equal(g3, np.array([1, 0, 0]))
        assert np.array_equal(g4, np.array([0, 1, 0]))

    def test_binary_tree_internal_node_output(self):
        ts = self.tree_internal_node()
        effect_size_df = self.tree_internal_node_df()
        sim_result = tstrait.sim_phenotype(
            ts=ts, effect_size=effect_size_df, h2=0.1, random_seed=1
        )
        assert np.array_equal(sim_result["GeneticValue"], np.array([3, 1, 2]))

    def non_binary_tree(self):
        # 2.00      7
        #     ┊ ┏━┏━━┏━┻━━┓   ┊
        # 1.00┊ ┃ ┃ ┃     6   ┊
        #     ┊ ┃ ┃ ┃  ┏━┳┻┓ ┊
        # 0.00 0 1 2   3 4 5
        ts = tskit.Tree.generate_balanced(6, arity=4, span=10).tree_sequence

        tables = ts.dump_tables()
        for j in range(3):
            tables.sites.add_row(j, "A")

        tables.individuals.add_row()
        tables.individuals.add_row()
        tables.individuals.add_row()
        individuals = tables.nodes.individual
        individuals[0] = 0
        individuals[1] = 0
        individuals[2] = 1
        individuals[3] = 1
        individuals[4] = 2
        individuals[5] = 2
        tables.nodes.individual = individuals

        tables.mutations.add_row(site=0, node=6, derived_state="T")
        tables.mutations.add_row(site=1, node=0, derived_state="T")
        tables.mutations.add_row(site=2, node=6, derived_state="T")
        tables.mutations.add_row(site=2, node=6, derived_state="C", parent=2)
        tables.mutations.add_row(site=2, node=5, derived_state="T", parent=3)

        ts = tables.tree_sequence()

        return ts

    def non_binary_tree_df(self):
        effect_size = {
            "SiteID": np.arange(3),
            "CausalState": ["T", "T", "C"],
            "EffectSize": np.ones(3),
        }
        effect_size_df = pd.DataFrame(effect_size)
        return effect_size_df

    def test_non_binary_tree_genotype(self):
        ts = self.non_binary_tree()
        effect_size_df = self.non_binary_tree_df()
        simulator = tstrait.PhenotypeSimulator(
            ts=ts, effect_size_df=effect_size_df, h2=0.1, random_seed=1
        )
        tree = ts.first()
        g1 = simulator._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g2 = simulator._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g3 = simulator._individual_genotype(tree, ts.site(2), "C", ts.num_nodes)
        assert np.array_equal(g1, np.array([0, 1, 2]))
        assert np.array_equal(g2, np.array([1, 0, 0]))
        assert np.array_equal(g3, np.array([0, 1, 1]))

    def test_non_binary_tree_output(self):
        ts = self.non_binary_tree()
        effect_size_df = self.non_binary_tree_df()
        sim_result = tstrait.sim_phenotype(
            ts=ts, effect_size=effect_size_df, h2=0.1, random_seed=1
        )
        assert np.array_equal(sim_result["GeneticValue"], np.array([1, 2, 3]))

    def multiple_node_tree(self):
        ts = tskit.Tree.generate_comb(6, span=10).tree_sequence
        tables = ts.dump_tables()
        for j in range(5):
            tables.sites.add_row(j, "A")

        tables.individuals.add_row()
        tables.individuals.add_row()
        individuals = tables.nodes.individual
        individuals[0] = 0
        individuals[1] = 1
        individuals[2] = 1
        individuals[3] = 1
        individuals[4] = 0
        individuals[5] = 0

        tables.nodes.individual = individuals

        tables.mutations.add_row(site=0, node=7, derived_state="T")
        tables.mutations.add_row(site=1, node=0, derived_state="T")
        tables.mutations.add_row(site=1, node=0, derived_state="A", parent=1)

        tables.mutations.add_row(site=2, node=9, derived_state="T")
        tables.mutations.add_row(site=2, node=3, derived_state="C", parent=3)

        tables.mutations.add_row(site=3, node=8, derived_state="T")
        tables.mutations.add_row(site=3, node=8, derived_state="A", parent=5)
        tables.mutations.add_row(site=3, node=8, derived_state="C", parent=6)
        tables.mutations.add_row(site=3, node=3, derived_state="A", parent=7)
        tables.mutations.add_row(site=3, node=6, derived_state="G", parent=7)

        tables.mutations.add_row(site=4, node=8, derived_state="A")

        ts = tables.tree_sequence()

        return ts

    def multiple_node_df(self):
        effect_size = {
            "SiteID": np.arange(5),
            "CausalState": ["T", "A", "T", "C", "A"],
            "EffectSize": np.ones(5) * 2,
        }
        effect_size_df = pd.DataFrame(effect_size)

        return effect_size_df

    def test_multiple_node_genotype(self):
        ts = self.multiple_node_tree()
        effect_size_df = self.multiple_node_df()
        simulator = tstrait.PhenotypeSimulator(
            ts=ts, effect_size_df=effect_size_df, h2=0.1, random_seed=1
        )
        tree = ts.first()
        g1 = simulator._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g2 = simulator._individual_genotype(tree, ts.site(1), "A", ts.num_nodes)
        g3 = simulator._individual_genotype(tree, ts.site(2), "T", ts.num_nodes)
        g4 = simulator._individual_genotype(tree, ts.site(3), "C", ts.num_nodes)
        g5 = simulator._individual_genotype(tree, ts.site(4), "A", ts.num_nodes)

        assert np.array_equal(g1, np.array([2, 1]))
        assert np.array_equal(g2, np.array([3, 3]))
        assert np.array_equal(g3, np.array([2, 2]))
        assert np.array_equal(g4, np.array([0, 1]))
        assert np.array_equal(g5, np.array([3, 3]))

    def test_multiple_node_output(self):
        ts = self.multiple_node_tree()
        effect_size_df = self.multiple_node_df()
        sim_result = tstrait.sim_phenotype(
            ts=ts, effect_size=effect_size_df, h2=0.1, random_seed=1
        )
        assert np.array_equal(sim_result["GeneticValue"], np.array([20, 20]))

    def individual_tree(self):
        ts = tskit.Tree.generate_comb(6, span=10).tree_sequence
        tables = ts.dump_tables()
        for j in range(5):
            tables.sites.add_row(j, "A")

        tables.individuals.add_row()
        tables.individuals.add_row()
        tables.individuals.add_row()
        tables.individuals.add_row()
        tables.individuals.add_row()
        tables.individuals.add_row()
        individuals = tables.nodes.individual
        individuals[0] = 0
        individuals[1] = 1
        individuals[2] = 2
        individuals[3] = 3
        individuals[4] = 4
        individuals[5] = 5

        tables.nodes.individual = individuals

        tables.mutations.add_row(site=0, node=7, derived_state="T")
        tables.mutations.add_row(site=1, node=0, derived_state="T")
        tables.mutations.add_row(site=1, node=0, derived_state="A", parent=1)

        tables.mutations.add_row(site=2, node=9, derived_state="T")
        tables.mutations.add_row(site=2, node=3, derived_state="C", parent=3)

        tables.mutations.add_row(site=3, node=8, derived_state="T")
        tables.mutations.add_row(site=3, node=8, derived_state="A", parent=5)
        tables.mutations.add_row(site=3, node=8, derived_state="C", parent=6)
        tables.mutations.add_row(site=3, node=3, derived_state="A", parent=7)
        tables.mutations.add_row(site=3, node=6, derived_state="G", parent=7)

        tables.mutations.add_row(site=4, node=8, derived_state="A")

        ts = tables.tree_sequence()

        return ts

    def individual_tree_df(self):
        effect_size = {
            "SiteID": np.arange(5),
            "CausalState": ["T", "A", "T", "C", "A"],
            "EffectSize": [1, 2, 3, 4, 5],
        }
        effect_size_df = pd.DataFrame(effect_size)
        return effect_size_df

    def test_individual_tree_genotype(self):
        ts = self.individual_tree()
        effect_size_df = self.individual_tree_df()
        simulator = tstrait.PhenotypeSimulator(
            ts=ts, effect_size_df=effect_size_df, h2=0.1, random_seed=1
        )
        tree = ts.first()
        g1 = simulator._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g2 = simulator._individual_genotype(tree, ts.site(1), "A", ts.num_nodes)
        g3 = simulator._individual_genotype(tree, ts.site(2), "T", ts.num_nodes)
        g4 = simulator._individual_genotype(tree, ts.site(3), "C", ts.num_nodes)
        g5 = simulator._individual_genotype(tree, ts.site(4), "A", ts.num_nodes)

        assert np.array_equal(g1, np.array([0, 0, 0, 1, 1, 1]))
        assert np.array_equal(g2, np.array([1, 1, 1, 1, 1, 1]))
        assert np.array_equal(g3, np.array([0, 1, 1, 0, 1, 1]))
        assert np.array_equal(g4, np.array([0, 0, 1, 0, 0, 0]))
        assert np.array_equal(g5, np.array([1, 1, 1, 1, 1, 1]))

    def test_individual_tree_output(self):
        ts = self.individual_tree()
        effect_size_df = self.individual_tree_df()
        sim_result = tstrait.sim_phenotype(
            ts=ts, effect_size=effect_size_df, h2=0.1, random_seed=1
        )
        assert np.array_equal(
            sim_result["GeneticValue"], np.array([7, 10, 14, 8, 11, 11])
        )


class Test_tree_sequence:
    def test_tree_sequence(self):
        ts = all_trees_ts(4)
        tables = ts.dump_tables()

        tables.individuals.add_row()
        tables.individuals.add_row()
        individuals = tables.nodes.individual
        individuals[0] = 0
        individuals[1] = 0
        individuals[2] = 1
        individuals[3] = 1
        tables.nodes.individual = individuals

        tables.sites.add_row(1, "A")
        tables.sites.add_row(7, "A")
        tables.sites.add_row(11, "G")
        tables.sites.add_row(12, "C")
        tables.sites.add_row(13, "A")
        tables.sites.add_row(14, "A")

        tables.mutations.add_row(site=0, node=4, derived_state="T")
        tables.mutations.add_row(site=1, node=4, derived_state="T")
        tables.mutations.add_row(site=2, node=5, derived_state="C")
        tables.mutations.add_row(site=2, node=4, derived_state="G", parent=2)
        tables.mutations.add_row(site=3, node=5, derived_state="C")
        tables.mutations.add_row(site=3, node=3, derived_state="T", parent=4)
        tables.mutations.add_row(site=4, node=3, derived_state="T")
        tables.mutations.add_row(site=4, node=3, derived_state="A", parent=6)
        tables.mutations.add_row(site=4, node=3, derived_state="C", parent=7)
        tables.mutations.add_row(site=5, node=4, derived_state="C")
        tables.mutations.add_row(site=5, node=4, derived_state="C", parent=9)
        tables.mutations.add_row(site=5, node=1, derived_state="A")

        ts = tables.tree_sequence()

        effect_size = {
            "SiteID": np.arange(6),
            "CausalState": np.array(["T", "T", "C", "T", "C", "C"]),
            "EffectSize": np.ones(6),
        }
        effect_size_df = pd.DataFrame(effect_size)

        sim_result = tstrait.sim_phenotype(
            ts=ts, effect_size=effect_size_df, h2=0.1, random_seed=1
        )
        assert np.array_equal(sim_result["GeneticValue"], np.array([2, 8]))
