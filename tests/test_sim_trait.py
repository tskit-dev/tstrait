import tstrait
import msprime
import numpy as np
import pytest
import tskit
import functools

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

class Test_sim_phenotype_output_dim:
    @pytest.mark.parametrize("num_ind", [1, 2, np.array([5])[0]])
    @pytest.mark.parametrize("num_causal", [1, 2, np.array([3])[0]])
    @pytest.mark.parametrize("alpha", [0, 1, -1.1])
    @pytest.mark.parametrize("random_seed", [1, 2, None])
    def test_output_dim_normal(self, num_ind, num_causal, alpha, random_seed):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        sim_result = tstrait.sim_traits(
            ts=ts, num_causal=num_causal, model=model, alpha=alpha,
            random_seed=random_seed
        )
        assert sim_result.shape[0] == num_causal
        assert sim_result.shape[1] == 3
        assert len(sim_result["SiteID"]) == num_causal
        assert len(sim_result["CausalState"]) == num_causal
        assert len(sim_result["EffectSize"]) == num_causal

    def test_binary_mutation_model(self):
        random_seed = 1
        num_ind = 10
        num_causal = 5
        alpha = 1
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(
            ts, rate=0.01, random_seed=random_seed, model="binary"
        )
        sim_result = tstrait.sim_traits(
            ts=ts, num_causal=num_causal, model=model, alpha=alpha,
            random_seed=random_seed
        )
        assert sim_result.shape[0] == num_causal
        assert sim_result.shape[1] == 3
        assert len(sim_result["SiteID"]) == num_causal
        assert len(sim_result["CausalState"]) == num_causal
        assert len(sim_result["EffectSize"]) == num_causal
    
    def test_output_dim_exponential(self):
        model = tstrait.trait_model(distribution="exponential", scale=1)        
        random_seed = 1
        num_ind = 10
        num_causal = 5
        alpha = 1
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        sim_result = tstrait.sim_traits(
            ts=ts, num_causal=num_causal, model=model, alpha=alpha,
            random_seed=random_seed
        )
        assert sim_result.shape[0] == num_causal
        assert sim_result.shape[1] == 3
        assert len(sim_result["SiteID"]) == num_causal
        assert len(sim_result["CausalState"]) == num_causal
        assert len(sim_result["EffectSize"]) == num_causal

    def test_output_dim_fixed(self):
        model = tstrait.trait_model(distribution="fixed", value=1)
        random_seed = 1
        num_ind = 10
        num_causal = 5
        alpha = 1
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        sim_result = tstrait.sim_traits(
            ts=ts, num_causal=num_causal, model=model, alpha=alpha,
            random_seed=random_seed
        )
        assert sim_result.shape[0] == num_causal
        assert sim_result.shape[1] == 3
        assert len(sim_result["SiteID"]) == num_causal
        assert len(sim_result["CausalState"]) == num_causal
        assert len(sim_result["EffectSize"]) == num_causal

    def test_output_dim_t(self):
        model = tstrait.trait_model(distribution="t", mean=0, var=1, df=1)
        random_seed = 1
        num_ind = 10
        num_causal = 5
        alpha = 1
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        sim_result = tstrait.sim_traits(
            ts=ts, num_causal=num_causal, model=model, alpha=alpha,
            random_seed=random_seed
        )
        assert sim_result.shape[0] == num_causal
        assert sim_result.shape[1] == 3
        assert len(sim_result["SiteID"]) == num_causal
        assert len(sim_result["CausalState"]) == num_causal
        assert len(sim_result["EffectSize"]) == num_causal
        
    def test_output_dim_gamma(self):
        model = tstrait.trait_model(distribution="gamma", shape=1, scale=1)
        random_seed = 1
        num_ind = 10
        num_causal = 5
        alpha = 1
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        sim_result = tstrait.sim_traits(
            ts=ts, num_causal=num_causal, model=model, alpha=alpha,
            random_seed=random_seed
        )
        assert sim_result.shape[0] == num_causal
        assert sim_result.shape[1] == 3
        assert len(sim_result["SiteID"]) == num_causal
        assert len(sim_result["CausalState"]) == num_causal
        assert len(sim_result["EffectSize"]) == num_causal
        
class Test_sim_phenotype_input:
    @pytest.mark.parametrize("ts", [0, "a", [1, 1]])
    def test_ts(self, ts):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        with pytest.raises(TypeError, match="Input must be a tree sequence data"):
            tstrait.sim_traits(
                ts=ts, num_causal=3, model=model, alpha=1, random_seed=1
                )

    @pytest.mark.parametrize("num_causal", ["1", "a", [1, 1]])
    def test_num_causal(self, num_causal):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        with pytest.raises(
            TypeError, match="Number of causal sites must be an integer"
        ):
            tstrait.sim_traits(
                ts=ts, num_causal=num_causal, model=model, alpha=1, random_seed=1
                )
            
    @pytest.mark.parametrize("num_causal", [-1, 1.8, -1.5, 0])
    def test_num_causal_positive(self, num_causal):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        with pytest.raises(
            ValueError, match="Number of causal sites must be a positive integer"
        ):
            tstrait.sim_traits(
                ts=ts, num_causal=num_causal, model=model, alpha=1, random_seed=1
                )

    @pytest.mark.parametrize("model", ["normal", 1, [None, 1, "a"]])
    def test_model(self, model):
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=2)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=2)
        with pytest.raises(
            TypeError, match="Trait model must be an instance of TraitModel"
        ):
            tstrait.sim_traits(
                ts=ts, num_causal=1, model=model, alpha=1, random_seed=1
                )        

    @pytest.mark.parametrize("num_ind", [1, 2, np.array([5])[0]])
    @pytest.mark.parametrize("num_causal", [1, 2, np.array([3])[0]])
    @pytest.mark.parametrize("alpha", [0,1])
    @pytest.mark.parametrize("random_seed", [1, 2])
    def test_error_mutation(self, num_ind, num_causal, alpha, random_seed):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        with pytest.raises(ValueError, match="No mutation in the provided data"):
            tstrait.sim_traits(ts=ts, num_causal=num_causal, model=model, alpha=alpha,
                               random_seed=random_seed)
            
    @pytest.mark.parametrize("num_ind", [1, 2, np.array([5])[0]])
    @pytest.mark.parametrize("num_causal_add", [1, 2, 3])
    @pytest.mark.parametrize("alpha", [0,1])
    @pytest.mark.parametrize("random_seed", [1, 2])
    def test_error_num_causal_add(self, num_ind, num_causal_add, alpha, random_seed):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        num_causal = ts.num_sites + num_causal_add
        with pytest.raises(ValueError, match="There are less number of sites in the "
                           "tree sequence than the inputted number of causal sites"):
            tstrait.sim_traits(ts=ts, num_causal=num_causal, model=model, alpha=alpha,
                               random_seed=random_seed)

    def test_output_num_causal(self):
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        ts = msprime.sim_ancestry(3, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        num_causal = ts.num_sites
        sim_result = tstrait.sim_traits(
            ts=ts, num_causal=num_causal, model=model, alpha=-1,
            random_seed=1
            )
        assert sim_result.shape[0] == num_causal
        
    @pytest.mark.parametrize("alpha", ["1", "a", [1, 1]])
    def test_alpha(self, alpha):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        with pytest.raises(
            TypeError, match="Alpha must be a number"
        ):
            tstrait.sim_traits(
                ts=ts, num_causal=3, model=model, alpha=alpha, random_seed=1
                )
            
class Test_site_genotypes:
    def test_binary_tree(self):
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
        
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        simulator = tstrait.EffectSizeSimulator(
            ts=ts, num_causal=1, model=model, alpha=0.3, random_seed=1
            )
        tree = ts.first()
        
        c1 = simulator._obtain_allele_count(tree, ts.site(0))
        c2 = simulator._obtain_allele_count(tree, ts.site(1))
        c3 = simulator._obtain_allele_count(tree, ts.site(2))
        c4 = simulator._obtain_allele_count(tree, ts.site(3))
        c5 = simulator._obtain_allele_count(tree, ts.site(4))
        c6 = simulator._obtain_allele_count(tree, ts.site(5))
        c7 = simulator._obtain_allele_count(tree, ts.site(6))
        c8 = simulator._obtain_allele_count(tree, ts.site(7))
        c9 = simulator._obtain_allele_count(tree, ts.site(8))
        c10 = simulator._obtain_allele_count(tree, ts.site(10))
        c11 = simulator._obtain_allele_count(tree, ts.site(11))
        
        assert c1 == {"T": 1}
        assert c2 == {"T": 2}
        assert c3 == {"T": 1}
        assert c4 == {"T": 3}
        assert c5 == {"T": 3}
        assert c6 == {"A": 4}
        assert c7 == {"C": 1}
        assert c8 == {"T": 1, "C": 2}
        assert c9 == {"T": 3}
        assert c10 == {"C": 4}
        assert c11 == {"C": 3, "T": 1}

    def test_binary_tree_different_individual(self):
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
        
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        simulator = tstrait.EffectSizeSimulator(
            ts=ts, num_causal=1, model=model, alpha=0.3, random_seed=1
            )
        tree = ts.first()

        c1 = simulator._obtain_allele_count(tree, ts.site(0))
        c2 = simulator._obtain_allele_count(tree, ts.site(1))
        c3 = simulator._obtain_allele_count(tree, ts.site(2))
        c4 = simulator._obtain_allele_count(tree, ts.site(3))
        c5 = simulator._obtain_allele_count(tree, ts.site(4))
        c6 = simulator._obtain_allele_count(tree, ts.site(5))
        c7 = simulator._obtain_allele_count(tree, ts.site(6))
        c8 = simulator._obtain_allele_count(tree, ts.site(7))
        c9 = simulator._obtain_allele_count(tree, ts.site(8))

        assert c1 == {"T": 1}
        assert c2 == {"T": 2}
        assert c3 == {"T": 1}
        assert c4 == {"T": 3}
        assert c5 == {"T": 3}
        assert c6 == {"A": 4}
        assert c7 == {"C": 1}
        assert c8 == {"T": 1, "C": 2}
        assert c9 == {"T": 3}
        
    def test_binary_tree_internal_node(self):
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

        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        simulator = tstrait.EffectSizeSimulator(
            ts=ts, num_causal=1, model=model, alpha=0.3, random_seed=1
            )
        tree = ts.first()

        c1 = simulator._obtain_allele_count(tree, ts.site(0))
        c2 = simulator._obtain_allele_count(tree, ts.site(1))
        c3 = simulator._obtain_allele_count(tree, ts.site(2))

        assert c1 == {"T": 1, "G": 2}
        assert c2 == {"T": 1, "G": 2}
        assert c3 == {"G": 1, "C": 1}
        
    def test_non_binary_tree(self):
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
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        simulator = tstrait.EffectSizeSimulator(
            ts=ts, num_causal=1, model=model, alpha=0.3, random_seed=1
            )
        tree = ts.first()

        c1 = simulator._obtain_allele_count(tree, ts.site(0))
        c2 = simulator._obtain_allele_count(tree, ts.site(1))
        c3 = simulator._obtain_allele_count(tree, ts.site(2))

        assert c1 == {"T": 3}
        assert c2 == {"T": 1}
        assert c3 == {"C": 2, "T": 1}

    def test_binary_tree_additional(self):
        ts = tskit.Tree.generate_comb(6, span=10).tree_sequence
        tables = ts.dump_tables()
        for j in range(5):
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
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        simulator = tstrait.EffectSizeSimulator(
            ts=ts, num_causal=1, model=model, alpha=0.3, random_seed=1
            )
        tree = ts.first()

        c1 = simulator._obtain_allele_count(tree, ts.site(0))
        c2 = simulator._obtain_allele_count(tree, ts.site(1))
        c3 = simulator._obtain_allele_count(tree, ts.site(2))
        c4 = simulator._obtain_allele_count(tree, ts.site(3))
        c5 = simulator._obtain_allele_count(tree, ts.site(4))

        assert c1 == {"T": 3}
        assert c2 == {"A": 6}
        assert c3 == {"T": 4, "C": 1}
        assert c4 == {"G": 2, "C": 1}
        assert c5 == {"A": 6}
        
def sim_tree_seq():
    """
    The following tree sequence will be generated:
      10
    ┏━┻┓
    ┃  9
    ┃ ┏┻━┓
    ┃ ┃   8
    ┃ ┃  ┏┻━┓
    ┃ ┃  ┃  7
    ┃ ┃  ┃ ┏┻━┓
    ┃ ┃  ┃ ┃  6
    ┃ ┃  ┃ ┃ ┏┻━┓
    0 1  2 3 4  5

    Individual 0: Node 0 and 1
    Individual 1: Node 2 and 3
    Individual 3: Node 4 and 5

    Site 0 Ancestral State: "A"
        Causal Mutation: Node 8
        Reverse Mutation: Node 4

    Site 1 Ancestral State: "A"
        Causal Mutation: Node 5
        
    Site 0 Genotype:
        [A, A, T, T, A, T]
        Ancestral state: A
        Causal allele freq: 0.5
        Individual 0: 0 causal
        Individual 1: 2 causal
        Individual 2: 1 causal

    Site 1 Genotype:
        [A, A, A, A, A, T]
        Ancestral state: A
        Causal allele freq: 1/6
        Individual 0: 0 causal
        Individual 1: 0 causal
        Individual 2: 1 causal

    Effect size distribution:

    SD Formula:
        trait_sd / sqrt(2) * [sqrt(2 * freq * (1 - freq))] ^ alpha
        sqrt(2) from 2 causal sites
    """
    ts = tskit.Tree.generate_comb(6, span=2).tree_sequence

    tables = ts.dump_tables()
    for j in range(2):
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
    tables.mutations.add_row(site=0, node=8, derived_state="T")
    tables.mutations.add_row(site=0, node=4, derived_state="A", parent=0)
    tables.mutations.add_row(site=1, node=5, derived_state="T")
    ts = tables.tree_sequence()
    return ts

def sim_tree_seq_freq(alpha):
    const1 = np.sqrt(pow(2 * 0.5 * (1 - 0.5), alpha))
    const2 = np.sqrt(pow(2 * 1/6 * (1 - 1/6), alpha))
    
    return const1, const2

class Test_fixed_effect_size:
    @pytest.mark.parametrize("value", [0, 1, -2])
    @pytest.mark.parametrize("alpha", [0, 1, -0.5])
    @pytest.mark.parametrize("random_seed", [1, 2, None])
    def test_output_dim_normal(self, value, alpha, random_seed):
        model = tstrait.trait_model(distribution="fixed", value=value)
        ts = sim_tree_seq()
        sim_result = tstrait.sim_traits(
            ts=ts, num_causal=2, model=model, alpha=alpha,
            random_seed=random_seed
        )
        const1, const2 = sim_tree_seq_freq(alpha)
        assert np.array_equal(sim_result["SiteID"], np.array([0, 1]))
        assert np.array_equal(sim_result["CausalState"], np.array(["T", "T"]))
        assert np.isclose(sim_result["EffectSize"][0], value*const1)
        assert np.isclose(sim_result["EffectSize"][1], value*const2)

class Test_allele_freq_one:
    @pytest.mark.parametrize("alpha", [0, 1, -0.5])
    def test_allele_freq_one(self, alpha):
        ts = tskit.Tree.generate_comb(6, span=2).tree_sequence
        tables = ts.dump_tables()
        for j in range(2):
            tables.sites.add_row(j, "A")
        tables.mutations.add_row(site=0, node=8, derived_state="T")
        tables.mutations.add_row(site=0, node=8, derived_state="A", parent=0)
        tables.mutations.add_row(site=1, node=5, derived_state="T")
        tables.mutations.add_row(site=1, node=5, derived_state="A", parent=2)
        ts = tables.tree_sequence()
        model = tstrait.trait_model(distribution="normal", mean=0, var=1)
        sim_result = tstrait.sim_traits(
            ts=ts, num_causal=2, model=model, alpha=alpha, random_seed=1
        )
        assert np.array_equal(sim_result["EffectSize"], np.array([0, 0]))
        
class Test_tree_sequence:
    def const(self, freq, alpha):
        value = np.sqrt(pow(2 * freq * (1 - freq), alpha))
        return value
    
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
        
        model = tstrait.trait_model(distribution="fixed", value=2)
        sim_result = tstrait.sim_traits(
            ts=ts, num_causal=6, model=model, alpha=-1,
            random_seed=1
        )
        assert np.array_equal(sim_result["SiteID"], np.array([0,1,2,3,4,5]))
        assert np.array_equal(sim_result["CausalState"], np.array(["T","T","C","T","C","C"]))
        assert np.isclose(sim_result["EffectSize"][0], 2 * self.const(freq=0.5, alpha=-1))
        assert np.isclose(sim_result["EffectSize"][1], 2 * self.const(freq=0.75, alpha=-1))
        assert np.isclose(sim_result["EffectSize"][2], 2 * self.const(freq=0.25, alpha=-1))
        assert np.isclose(sim_result["EffectSize"][3], 2 * self.const(freq=0.25, alpha=-1))
        assert np.isclose(sim_result["EffectSize"][4], 2 * self.const(freq=0.25, alpha=-1))
        assert np.isclose(sim_result["EffectSize"][5], 2 * self.const(freq=0.5, alpha=-1))
        