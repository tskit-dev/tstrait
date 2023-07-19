import functools

import msprime
import numba
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


class Test_output:
    @pytest.mark.parametrize("num_ind", [1, 2, np.array([5])[0]])
    def test_output_single(self, num_ind):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        trait_df = tstrait.sim_trait(
            ts=ts,
            num_causal=5,
            model=model,
            alpha=0,
            random_seed=1,
        )
        genetic_df = tstrait.calculate_genetic_value(ts, trait_df)

        assert len(genetic_df) == num_ind
        assert genetic_df.shape[1] == 3
        assert np.array_equal(genetic_df["individual_id"], np.arange(num_ind))
        assert len(genetic_df["genetic_value"]) == num_ind
        assert np.array_equal(genetic_df["trait_id"], np.zeros(num_ind))

    @pytest.mark.parametrize("num_ind", [1, 2, np.array([5])[0]])
    def test_output_multiple(self, num_ind):
        mean = np.array([0, 0, 0])
        var = np.array([1, 9, 16])
        cor = np.array([[1, 0.1, 0.2], [0.1, 1, 0.3], [0.2, 0.3, 1]])
        model = tstrait.trait_model(
            distribution="multi_normal", mean=mean, var=var, cor=cor
        )
        ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        trait_df = tstrait.sim_trait(
            ts=ts,
            num_causal=5,
            model=model,
            alpha=0,
            random_seed=1,
        )
        genetic_df = tstrait.calculate_genetic_value(ts, trait_df)

        assert len(genetic_df) == num_ind * 3
        assert genetic_df.shape[1] == 3
        assert len(genetic_df["individual_id"]) == num_ind * 3
        assert len(genetic_df["genetic_value"]) == num_ind * 3
        assert len(genetic_df["trait_id"]) == num_ind * 3

    @pytest.mark.parametrize("num_ind", [1, 2, np.array([5])[0]])
    def test_binary_mutation_model(self, num_ind):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1, model="binary")
        trait_df = tstrait.sim_trait(
            ts=ts,
            num_causal=5,
            model=model,
            alpha=0,
            random_seed=1,
        )
        genetic_df = tstrait.calculate_genetic_value(ts, trait_df)

        assert len(genetic_df) == num_ind
        assert genetic_df.shape[1] == 3
        assert np.array_equal(genetic_df["individual_id"], np.arange(num_ind))
        assert len(genetic_df["genetic_value"]) == num_ind
        assert np.array_equal(genetic_df["trait_id"], np.zeros(num_ind))


class Test_genetic_value_input:
    @pytest.mark.parametrize("ts", [0, "a", [1, 1]])
    def test_ts(self, ts):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        ts1 = msprime.sim_ancestry(10, sequence_length=100_000, random_seed=1)
        ts1 = msprime.sim_mutations(ts1, rate=0.01, random_seed=1)
        trait_df = tstrait.sim_trait(
            ts=ts1,
            num_causal=5,
            model=model,
            alpha=0,
            random_seed=1,
        )
        with pytest.raises(TypeError, match="Input must be a tree sequence data"):
            tstrait.calculate_genetic_value(ts, trait_df)

    def test_df(self):
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        with pytest.raises(TypeError, match="trait_df must be a pandas dataframe"):
            tstrait.calculate_genetic_value(ts, [0, 1])

    def test_df_site_id(self):
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        df = pd.DataFrame(
            {
                "siteID": [0, 1],
                "causal_state": ["A", "A"],
                "effect_size": [0.1, 0.1],
                "trait_id": [0, 1],
            }
        )
        with pytest.raises(
            ValueError, match="site_id is not included in the trait dataframe"
        ):
            tstrait.calculate_genetic_value(ts, df)

    def test_df_causal_state(self):
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        df = pd.DataFrame(
            {
                "site_id": [0, 1],
                "causal_te": ["A", "A"],
                "effect_size": [0.1, 0.1],
                "trait_id": [0, 1],
            }
        )
        with pytest.raises(
            ValueError, match="causal_state is not included in the trait dataframe"
        ):
            tstrait.calculate_genetic_value(ts, df)

    def test_df_effect_size(self):
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        df = pd.DataFrame(
            {
                "site_id": [0, 1],
                "causal_state": ["A", "A"],
                "Effect": [0.1, 0.1],
                "trait_id": [0, 1],
            }
        )
        with pytest.raises(
            ValueError, match="effect_size is not included in the trait dataframe"
        ):
            tstrait.calculate_genetic_value(ts, df)

    def test_df_trait_id(self):
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        df = pd.DataFrame(
            {
                "site_id": [0, 1],
                "causal_state": ["A", "A"],
                "effect_size": [0.1, 0.1],
                "traitID": [0, 1],
            }
        )
        with pytest.raises(
            ValueError, match="trait_id is not included in the trait dataframe"
        ):
            tstrait.calculate_genetic_value(ts, df)


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
            "site_id": np.arange(12),
            "causal_state": [
                "T",
                "T",
                "T",
                "T",
                "T",
                "A",
                "C",
                "T",
                "T",
                "A",
                "C",
                "C",
            ],
            "effect_size": np.ones(12),
            "trait_id": np.zeros(12),
        }
        effect_size_df = pd.DataFrame(effect_size)

        return effect_size_df

    def binary_tree_df_multiple(self):
        effect_size = {
            "site_id": np.arange(12),
            "causal_state": [
                "T",
                "T",
                "T",
                "T",
                "T",
                "A",
                "C",
                "T",
                "T",
                "A",
                "C",
                "C",
            ],
            "effect_size": np.ones(12) * 2,
            "trait_id": np.ones(12),
        }
        effect_size_df = pd.DataFrame(effect_size)
        df = pd.concat([self.binary_tree_df(), effect_size_df])

        return df

    def test_binary_tree_genotype(self):
        ts = self.binary_tree()
        trait_df = self.binary_tree_df()
        genetic = tstrait.GeneticValue(ts=ts, trait_df=trait_df)
        tree = ts.first()

        g0 = genetic._individual_genotype(tree, ts.site(0), "G", ts.num_nodes)
        g1 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g2 = genetic._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g3 = genetic._individual_genotype(tree, ts.site(2), "T", ts.num_nodes)
        g4 = genetic._individual_genotype(tree, ts.site(3), "T", ts.num_nodes)
        g5 = genetic._individual_genotype(tree, ts.site(4), "T", ts.num_nodes)
        g6 = genetic._individual_genotype(tree, ts.site(5), "A", ts.num_nodes)
        g7 = genetic._individual_genotype(tree, ts.site(6), "C", ts.num_nodes)
        g8 = genetic._individual_genotype(tree, ts.site(7), "T", ts.num_nodes)
        g9 = genetic._individual_genotype(tree, ts.site(8), "T", ts.num_nodes)
        g10 = genetic._individual_genotype(tree, ts.site(9), "A", ts.num_nodes)
        g11 = genetic._individual_genotype(tree, ts.site(10), "C", ts.num_nodes)
        g12 = genetic._individual_genotype(tree, ts.site(11), "C", ts.num_nodes)

        assert np.array_equal(g0, np.array([0, 0]))
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

    def test_binary_tree_numba_func(self):
        ts = self.binary_tree()
        num_nodes = ts.num_nodes
        has_mutation = np.zeros(num_nodes + 1, dtype=bool)
        has_mutation[0] = True
        tree = ts.first()
        stack = numba.typed.List([0])
        genotype = tstrait.genetic_value._traversal_genotype(
            nodes_individual=ts.nodes_individual,
            left_child_array=tree.left_child_array,
            right_sib_array=tree.right_sib_array,
            stack=stack,
            has_mutation=has_mutation,
            num_individuals=ts.num_individuals,
            num_nodes=num_nodes,
        )
        assert np.array_equal(genotype, np.array([1, 0]))

    def test_binary_tree_output(self):
        ts = self.binary_tree()
        trait_df = self.binary_tree_df()
        genetic_df = tstrait.calculate_genetic_value(ts=ts, trait_df=trait_df)
        assert np.array_equal(genetic_df["genetic_value"], np.array([14, 16]))

    def test_binary_tree_output_multiple(self):
        ts = self.binary_tree()
        trait_df = self.binary_tree_df_multiple()
        genetic_df = tstrait.calculate_genetic_value(ts=ts, trait_df=trait_df)
        data = {
            "individual_id": [0, 1, 0, 1],
            "genetic_value": [14, 16, 28, 32],
            "trait_id": [0, 0, 1, 1],
        }
        df = pd.DataFrame(data)
        pd.testing.assert_frame_equal(df, genetic_df, check_dtype=False)

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
            "site_id": np.arange(9),
            "causal_state": ["T", "T", "T", "T", "T", "A", "C", "T", "T"],
            "effect_size": np.ones(9),
            "trait_id": np.zeros(9),
        }
        effect_size_df = pd.DataFrame(effect_size)
        return effect_size_df

    def binary_tree_different_individual_multiple_df(self):
        effect_size = {
            "site_id": np.arange(9),
            "causal_state": ["T", "T", "T", "T", "T", "A", "C", "T", "T"],
            "effect_size": np.ones(9) * 2,
            "trait_id": np.ones(9),
        }
        effect_size_df = pd.DataFrame(effect_size)
        df = pd.concat([self.binary_tree_different_individual_df(), effect_size_df])
        effect_size = {
            "site_id": np.arange(9),
            "causal_state": ["T", "T", "T", "T", "T", "A", "C", "T", "T"],
            "effect_size": np.ones(9) * 3,
            "trait_id": np.ones(9) * 2,
        }
        effect_size_df = pd.DataFrame(effect_size)
        df = pd.concat([df, effect_size_df])

        df = df.reset_index()
        del df["index"]

        return df

    def test_binary_tree_different_individual_genotype(self):
        ts = self.binary_tree_different_individual()
        trait_df = self.binary_tree_different_individual_df()
        genetic = tstrait.GeneticValue(ts=ts, trait_df=trait_df)
        tree = ts.first()

        g1 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g2 = genetic._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g3 = genetic._individual_genotype(tree, ts.site(2), "T", ts.num_nodes)
        g4 = genetic._individual_genotype(tree, ts.site(3), "T", ts.num_nodes)
        g5 = genetic._individual_genotype(tree, ts.site(4), "T", ts.num_nodes)
        g6 = genetic._individual_genotype(tree, ts.site(5), "A", ts.num_nodes)
        g7 = genetic._individual_genotype(tree, ts.site(6), "C", ts.num_nodes)
        g8 = genetic._individual_genotype(tree, ts.site(7), "T", ts.num_nodes)
        g9 = genetic._individual_genotype(tree, ts.site(8), "T", ts.num_nodes)

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
        trait_df = self.binary_tree_different_individual_df()
        genetic_df = tstrait.calculate_genetic_value(ts=ts, trait_df=trait_df)
        assert np.array_equal(genetic_df["genetic_value"], np.array([9, 10]))

    def test_binary_tree_output_multiple_different_individual(self):
        ts = self.binary_tree_different_individual()
        trait_df = self.binary_tree_different_individual_multiple_df()
        genetic_df = tstrait.calculate_genetic_value(ts=ts, trait_df=trait_df)
        data = {
            "individual_id": [0, 1, 0, 1, 0, 1],
            "genetic_value": [9, 10, 18, 20, 27, 30],
            "trait_id": [0, 0, 1, 1, 2, 2],
        }
        df = pd.DataFrame(data)
        pd.testing.assert_frame_equal(df, genetic_df, check_dtype=False)

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
            "site_id": np.arange(3),
            "causal_state": ["G", "T", "T"],
            "effect_size": np.ones(3),
            "trait_id": np.zeros(3),
        }
        effect_size_df = pd.DataFrame(effect_size)
        return effect_size_df

    def tree_internal_node_multiple_df(self):
        effect_size = {
            "site_id": [2, 2],
            "causal_state": ["T", "G"],
            "effect_size": [1, 2],
            "trait_id": [0, 1],
        }
        effect_size_df = pd.DataFrame(effect_size)
        return effect_size_df

    def test_binary_tree_internal_node_genotype(self):
        ts = self.tree_internal_node()
        trait_df = self.tree_internal_node_df()
        genetic = tstrait.GeneticValue(ts=ts, trait_df=trait_df)
        tree = ts.first()
        g1 = genetic._individual_genotype(tree, ts.site(0), "G", ts.num_nodes)
        g2 = genetic._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g3 = genetic._individual_genotype(tree, ts.site(2), "T", ts.num_nodes)
        g4 = genetic._individual_genotype(tree, ts.site(2), "G", ts.num_nodes)

        assert np.array_equal(g1, np.array([1, 0, 2]))
        assert np.array_equal(g2, np.array([1, 1, 0]))
        assert np.array_equal(g3, np.array([1, 0, 0]))
        assert np.array_equal(g4, np.array([0, 1, 0]))

    def test_binary_tree_internal_node_output(self):
        ts = self.tree_internal_node()
        trait_df = self.tree_internal_node_df()
        genetic_df = tstrait.calculate_genetic_value(ts=ts, trait_df=trait_df)
        assert np.array_equal(genetic_df["genetic_value"], np.array([3, 1, 2]))

    def test_different_causal_site(self):
        ts = self.tree_internal_node()
        trait_df = self.tree_internal_node_multiple_df()
        genetic_df = tstrait.calculate_genetic_value(ts=ts, trait_df=trait_df)
        data = {
            "individual_id": [0, 1, 2, 0, 1, 2],
            "genetic_value": [1, 0, 0, 0, 2, 0],
            "trait_id": [0, 0, 0, 1, 1, 1],
        }
        df = pd.DataFrame(data)
        pd.testing.assert_frame_equal(df, genetic_df, check_dtype=False)

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
            "site_id": np.arange(3),
            "causal_state": ["T", "T", "C"],
            "effect_size": np.ones(3),
            "trait_id": np.zeros(3),
        }
        effect_size_df = pd.DataFrame(effect_size)
        return effect_size_df

    def test_non_binary_tree_genotype(self):
        ts = self.non_binary_tree()
        trait_df = self.non_binary_tree_df()
        genetic = tstrait.GeneticValue(ts=ts, trait_df=trait_df)
        tree = ts.first()
        g1 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g2 = genetic._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g3 = genetic._individual_genotype(tree, ts.site(2), "C", ts.num_nodes)
        assert np.array_equal(g1, np.array([0, 1, 2]))
        assert np.array_equal(g2, np.array([1, 0, 0]))
        assert np.array_equal(g3, np.array([0, 1, 1]))

    def test_non_binary_tree_output(self):
        ts = self.non_binary_tree()
        trait_df = self.non_binary_tree_df()
        genetic_df = tstrait.calculate_genetic_value(ts=ts, trait_df=trait_df)
        assert np.array_equal(genetic_df["genetic_value"], np.array([1, 2, 3]))

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
            "site_id": np.arange(5),
            "causal_state": ["T", "A", "T", "C", "A"],
            "effect_size": np.ones(5) * 2,
            "trait_id": np.zeros(5),
        }
        effect_size_df = pd.DataFrame(effect_size)

        return effect_size_df

    def test_multiple_node_genotype(self):
        ts = self.multiple_node_tree()
        trait_df = self.multiple_node_df()
        genetic = tstrait.GeneticValue(ts=ts, trait_df=trait_df)
        tree = ts.first()
        g1 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g2 = genetic._individual_genotype(tree, ts.site(1), "A", ts.num_nodes)
        g3 = genetic._individual_genotype(tree, ts.site(2), "T", ts.num_nodes)
        g4 = genetic._individual_genotype(tree, ts.site(3), "C", ts.num_nodes)
        g5 = genetic._individual_genotype(tree, ts.site(4), "A", ts.num_nodes)

        assert np.array_equal(g1, np.array([2, 1]))
        assert np.array_equal(g2, np.array([3, 3]))
        assert np.array_equal(g3, np.array([2, 2]))
        assert np.array_equal(g4, np.array([0, 1]))
        assert np.array_equal(g5, np.array([3, 3]))

    def test_multiple_node_output(self):
        ts = self.multiple_node_tree()
        trait_df = self.multiple_node_df()
        genetic_df = tstrait.calculate_genetic_value(ts=ts, trait_df=trait_df)
        assert np.array_equal(genetic_df["genetic_value"], np.array([20, 20]))

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
            "site_id": np.arange(5),
            "causal_state": ["T", "A", "T", "C", "A"],
            "effect_size": [1, 2, 3, 4, 5],
            "trait_id": np.ones(5),
        }
        effect_size_df = pd.DataFrame(effect_size)
        return effect_size_df

    def test_individual_tree_genotype(self):
        ts = self.individual_tree()
        trait_df = self.individual_tree_df()
        genetic = tstrait.GeneticValue(ts=ts, trait_df=trait_df)
        tree = ts.first()
        g1 = genetic._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g2 = genetic._individual_genotype(tree, ts.site(1), "A", ts.num_nodes)
        g3 = genetic._individual_genotype(tree, ts.site(2), "T", ts.num_nodes)
        g4 = genetic._individual_genotype(tree, ts.site(3), "C", ts.num_nodes)
        g5 = genetic._individual_genotype(tree, ts.site(4), "A", ts.num_nodes)

        assert np.array_equal(g1, np.array([0, 0, 0, 1, 1, 1]))
        assert np.array_equal(g2, np.array([1, 1, 1, 1, 1, 1]))
        assert np.array_equal(g3, np.array([0, 1, 1, 0, 1, 1]))
        assert np.array_equal(g4, np.array([0, 0, 1, 0, 0, 0]))
        assert np.array_equal(g5, np.array([1, 1, 1, 1, 1, 1]))

    def test_individual_tree_output(self):
        ts = self.individual_tree()
        trait_df = self.individual_tree_df()
        genetic_df = tstrait.calculate_genetic_value(ts=ts, trait_df=trait_df)
        assert np.array_equal(
            genetic_df["genetic_value"], np.array([7, 10, 14, 8, 11, 11])
        )
        assert np.array_equal(genetic_df["trait_id"], np.ones(6))


class Test_tree_sequence:
    def sample_tree_sequence(self):
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
        tables.sites.add_row(15, "A")
        tables.sites.add_row(16, "A")

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
        tables.mutations.add_row(site=6, node=0, derived_state="C")
        tables.mutations.add_row(site=6, node=2, derived_state="T")
        tables.mutations.add_row(site=7, node=0, derived_state="T")
        tables.mutations.add_row(site=7, node=2, derived_state="G")

        ts = tables.tree_sequence()

        return ts

    def test_tree_sequence(self):
        ts = self.sample_tree_sequence()
        effect_size = {
            "site_id": np.arange(6),
            "causal_state": np.array(["T", "T", "C", "T", "C", "C"]),
            "effect_size": np.ones(6),
            "trait_id": np.zeros(6),
        }
        trait_df = pd.DataFrame(effect_size)
        genetic_df = tstrait.calculate_genetic_value(ts=ts, trait_df=trait_df)
        assert np.array_equal(genetic_df["genetic_value"], np.array([2, 8]))

    def test_tree_sequence_multiple(self):
        ts = self.sample_tree_sequence()
        effect_size = {
            "site_id": [6, 7, 6, 7, 0],
            "causal_state": np.array(["C", "T", "T", "T", "A"]),
            "effect_size": [1, 1, 1, 1, 1],
            "trait_id": [0, 0, 1, 1, 2],
        }
        trait_df = pd.DataFrame(effect_size)
        genetic_df = tstrait.calculate_genetic_value(ts=ts, trait_df=trait_df)
        data = {
            "individual_id": [0, 1, 0, 1, 0, 1],
            "genetic_value": [2, 0, 1, 1, 2, 0],
            "trait_id": [0, 0, 1, 1, 2, 2],
        }
        df = pd.DataFrame(data)
        pd.testing.assert_frame_equal(df, genetic_df, check_dtype=False)
