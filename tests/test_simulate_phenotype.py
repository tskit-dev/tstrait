import pytest
import msprime
import numpy as np
import tskit
import tstrait.simulate_phenotype as simulate_phenotype
import tstrait.trait_model as trait_model
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
    @pytest.mark.parametrize("h2", [0.1, np.array([0.5])[0]])
    @pytest.mark.parametrize("random_seed", [1, 2])
    def test_output_dim_additive(self, num_ind, num_causal, h2, random_seed):
        model = trait_model.TraitModelAdditive(0, 1)
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(
            ts, num_causal, model, h2, random_seed
        )

        assert len(phenotype_result.__dict__) == 4
        assert len(genetic_result.__dict__) == 4

        assert len(phenotype_result.individual_id) == num_ind
        assert len(phenotype_result.phenotype) == num_ind
        assert len(phenotype_result.environment_noise) == num_ind
        assert len(phenotype_result.genetic_value) == num_ind

        assert len(genetic_result.site_id) == num_causal
        assert len(genetic_result.causal_allele) == num_causal
        assert len(genetic_result.effect_size) == num_causal
        assert len(genetic_result.allele_frequency) == num_causal

    def test_binary_mutation_model(self):
        random_seed = 1
        num_ind = 10
        num_causal = 5
        h2 = 0.3
        model = trait_model.TraitModelAdditive(0, 1)
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(
            ts, rate=0.01, random_seed=random_seed, model="binary"
        )
        phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(
            ts, num_causal, model, h2, random_seed
        )

        assert len(phenotype_result.__dict__) == 4
        assert len(genetic_result.__dict__) == 4

        assert len(phenotype_result.individual_id) == num_ind
        assert len(phenotype_result.phenotype) == num_ind
        assert len(phenotype_result.environment_noise) == num_ind
        assert len(phenotype_result.genetic_value) == num_ind

        assert len(genetic_result.site_id) == num_causal
        assert len(genetic_result.causal_allele) == num_causal
        assert len(genetic_result.effect_size) == num_causal
        assert len(genetic_result.allele_frequency) == num_causal

    @pytest.mark.parametrize("num_ind", [1, 2, 5])
    @pytest.mark.parametrize("num_causal", [1, 2, 3])
    @pytest.mark.parametrize("h2", [0.1, 0.5])
    @pytest.mark.parametrize("random_seed", [1, 2])
    def test_output_dim_Allele(self, num_ind, num_causal, h2, random_seed):
        model = trait_model.TraitModelAlleleFrequency(0, 1, -1)
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(
            ts, num_causal, model, h2, random_seed
        )

        assert len(phenotype_result.__dict__) == 4
        assert len(genetic_result.__dict__) == 4

        assert len(phenotype_result.individual_id) == num_ind
        assert len(phenotype_result.phenotype) == num_ind
        assert len(phenotype_result.environment_noise) == num_ind
        assert len(phenotype_result.genetic_value) == num_ind

        assert len(genetic_result.site_id) == num_causal
        assert len(genetic_result.causal_allele) == num_causal
        assert len(genetic_result.effect_size) == num_causal
        assert len(genetic_result.allele_frequency) == num_causal

        assert max(genetic_result.allele_frequency < 1) and min(
            genetic_result.allele_frequency > 0
        )


class Test_sim_phenotype_input:
    @pytest.mark.parametrize("ts", [0, "a", [1, 1]])
    def test_ts(self, ts):
        model = trait_model.TraitModelAdditive(0, 1)
        with pytest.raises(TypeError, match="Input should be a tree sequence data"):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(
                ts, 2, model, 0.3, 1
            )

    @pytest.mark.parametrize("num_ind", [1, 2, 5])
    @pytest.mark.parametrize("num_causal", [1, 2, 3])
    @pytest.mark.parametrize("h2", [0.1, 0.5])
    @pytest.mark.parametrize("random_seed", [1, 2])
    def test_no_mutation(self, num_ind, num_causal, h2, random_seed):
        model = trait_model.TraitModelAdditive(0, 1)
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        with pytest.raises(ValueError, match="No mutation in the provided data"):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(
                ts, num_causal, model, h2, random_seed
            )

    @pytest.mark.parametrize("model", [None, 1, "a"])
    def test_model(self, model):
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=2)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=2)
        with pytest.raises(
            TypeError, match="Trait model must be an instance of TraitModel"
        ):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(
                ts, 3, model, 0.3, 2
            )

    @pytest.mark.parametrize("h2", ["0", "a", [1, 1]])
    def test_h2(self, h2):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        model = trait_model.TraitModelAdditive(0, 1)
        with pytest.raises(TypeError, match="Heritability should be a number"):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(
                ts, 2, model, h2, 1
            )

    @pytest.mark.parametrize("h2", [-1, -0.1, 1.01])
    def test_h2_value(self, h2):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        model = trait_model.TraitModelAdditive(0, 1)
        with pytest.raises(ValueError, match="Heritability should be 0 <= h2 <= 1"):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(
                ts, 2, model, h2, 1
            )

    @pytest.mark.parametrize("num_ind", [1, 2, 5])
    @pytest.mark.parametrize("num_causal", [1, 2, 3])
    @pytest.mark.parametrize("random_seed", [1, 2])
    def test_h2_zero(self, num_ind, num_causal, random_seed):
        model = trait_model.TraitModelAdditive(0, 1)
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(
            ts, num_causal, model, 0, random_seed
        )

        assert np.allclose(
            phenotype_result.phenotype, phenotype_result.environment_noise
        )

    @pytest.mark.parametrize("num_ind", [1, 2, 5])
    @pytest.mark.parametrize("num_causal", [1, 2, 3])
    @pytest.mark.parametrize("random_seed", [1, 2])
    def test_h2_one(self, num_ind, num_causal, random_seed):
        model = trait_model.TraitModelAdditive(0, 1)
        ts = msprime.sim_ancestry(
            num_ind, sequence_length=100_000, random_seed=random_seed
        )
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(
            ts, num_causal, model, 1, random_seed
        )

        assert np.allclose(phenotype_result.phenotype, phenotype_result.genetic_value)
        assert np.array_equiv(phenotype_result.environment_noise, np.zeros(num_ind))

    @pytest.mark.parametrize("num_causal", ["1", "a", [1, 1]])
    def test_num_causal(self, num_causal):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        model = trait_model.TraitModelAdditive(0, 1)
        with pytest.raises(
            TypeError, match="Number of causal sites should be an integer"
        ):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(
                ts, num_causal, model, 0.3, 1
            )

    @pytest.mark.parametrize("num_causal", [-1, 1.8, -1.5, 0])
    def test_num_causal_value(self, num_causal):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        model = trait_model.TraitModelAdditive(0, 1)
        with pytest.raises(
            ValueError, match="Number of causal sites should be a positive integer"
        ):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(
                ts, num_causal, model, 0.3, 1
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

        model = trait_model.TraitModelAdditive(0, 1)
        rng = np.random.default_rng(1)
        simulator = simulate_phenotype.PhenotypeSimulator(ts, 1, 0.3, model, rng)
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

        c1 = simulator._obtain_allele_frequency(tree, ts.site(0))
        c2 = simulator._obtain_allele_frequency(tree, ts.site(1))
        c3 = simulator._obtain_allele_frequency(tree, ts.site(2))
        c4 = simulator._obtain_allele_frequency(tree, ts.site(3))
        c5 = simulator._obtain_allele_frequency(tree, ts.site(4))
        c6 = simulator._obtain_allele_frequency(tree, ts.site(5))
        c7 = simulator._obtain_allele_frequency(tree, ts.site(6))
        c8 = simulator._obtain_allele_frequency(tree, ts.site(7))
        c9 = simulator._obtain_allele_frequency(tree, ts.site(8))
        c10 = simulator._obtain_allele_frequency(tree, ts.site(10))
        c11 = simulator._obtain_allele_frequency(tree, ts.site(11))

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

        model = trait_model.TraitModelAdditive(0, 1)
        rng = np.random.default_rng(1)
        simulator = simulate_phenotype.PhenotypeSimulator(ts, 1, 0.3, model, rng)
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

        c1 = simulator._obtain_allele_frequency(tree, ts.site(0))
        c2 = simulator._obtain_allele_frequency(tree, ts.site(1))
        c3 = simulator._obtain_allele_frequency(tree, ts.site(2))
        c4 = simulator._obtain_allele_frequency(tree, ts.site(3))
        c5 = simulator._obtain_allele_frequency(tree, ts.site(4))
        c6 = simulator._obtain_allele_frequency(tree, ts.site(5))
        c7 = simulator._obtain_allele_frequency(tree, ts.site(6))
        c8 = simulator._obtain_allele_frequency(tree, ts.site(7))
        c9 = simulator._obtain_allele_frequency(tree, ts.site(8))

        assert np.array_equal(g1, np.array([1, 0]))
        assert np.array_equal(g2, np.array([1, 1]))
        assert np.array_equal(g3, np.array([0, 1]))
        assert np.array_equal(g4, np.array([1, 2]))
        assert np.array_equal(g5, np.array([2, 1]))
        assert np.array_equal(g6, np.array([2, 2]))
        assert np.array_equal(g7, np.array([1, 0]))
        assert np.array_equal(g8, np.array([0, 1]))
        assert np.array_equal(g9, np.array([1, 2]))

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

        # flags = tables.nodes.flags
        # Set nodes to be samples
        # flags[:] = 0
        # flags[:6] = tskit.NODE_IS_SAMPLE

        # tables.nodes.flags = flags

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

        model = trait_model.TraitModelAdditive(0, 1)
        rng = np.random.default_rng(1)
        simulator = simulate_phenotype.PhenotypeSimulator(ts, 1, 0.3, model, rng)
        tree = ts.first()

        g1 = simulator._individual_genotype(tree, ts.site(0), "G", ts.num_nodes)
        g2 = simulator._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g3 = simulator._individual_genotype(tree, ts.site(2), "T", ts.num_nodes)
        g4 = simulator._individual_genotype(tree, ts.site(2), "G", ts.num_nodes)

        assert np.array_equal(g1, np.array([1, 0, 2]))
        assert np.array_equal(g2, np.array([1, 1, 0]))
        assert np.array_equal(g3, np.array([1, 0, 0]))
        assert np.array_equal(g4, np.array([0, 1, 0]))

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
        model = trait_model.TraitModelAdditive(0, 1)
        rng = np.random.default_rng(1)
        simulator = simulate_phenotype.PhenotypeSimulator(ts, 1, 0.3, model, rng)
        tree = ts.first()

        g1 = simulator._individual_genotype(tree, ts.site(0), "T", ts.num_nodes)
        g2 = simulator._individual_genotype(tree, ts.site(1), "T", ts.num_nodes)
        g3 = simulator._individual_genotype(tree, ts.site(2), "C", ts.num_nodes)

        c1 = simulator._obtain_allele_frequency(tree, ts.site(0))
        c2 = simulator._obtain_allele_frequency(tree, ts.site(1))
        c3 = simulator._obtain_allele_frequency(tree, ts.site(2))

        assert np.array_equal(g1, np.array([0, 1, 2]))
        assert np.array_equal(g2, np.array([1, 0, 0]))
        assert np.array_equal(g3, np.array([0, 1, 1]))

        assert c1 == {"T": 3}
        assert c2 == {"T": 1}
        assert c3 == {"C": 2, "T": 1}

    def test_multiple_node_individual(self):
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

        model = trait_model.TraitModelAdditive(0, 1)
        rng = np.random.default_rng(1)
        simulator = simulate_phenotype.PhenotypeSimulator(ts, 1, 0.3, model, rng)
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

    def test_individual_single_node(self):
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

        model = trait_model.TraitModelAdditive(0, 1)
        rng = np.random.default_rng(1)
        simulator = simulate_phenotype.PhenotypeSimulator(ts, 1, 0.3, model, rng)
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


class Test_obtain_allele_frequency:
    def test_binary_tree(self):
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

        model = trait_model.TraitModelAdditive(0, 1)
        rng = np.random.default_rng(1)
        simulator = simulate_phenotype.PhenotypeSimulator(ts, 1, 0.3, model, rng)
        tree = ts.first()

        g1 = simulator._obtain_allele_frequency(tree, ts.site(0))
        g2 = simulator._obtain_allele_frequency(tree, ts.site(1))
        g3 = simulator._obtain_allele_frequency(tree, ts.site(2))
        g4 = simulator._obtain_allele_frequency(tree, ts.site(3))
        g5 = simulator._obtain_allele_frequency(tree, ts.site(4))

        assert g1 == {"T": 3}
        assert g2 == {"A": 6}
        assert g3 == {"T": 4, "C": 1}
        assert g4 == {"G": 2, "C": 1}
        assert g5 == {"A": 6}


class Test_sim_genetic_value:
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
        tables.mutations.add_row(site=2, node=4, derived_state="G")
        tables.mutations.add_row(site=3, node=5, derived_state="C")
        tables.mutations.add_row(site=3, node=3, derived_state="T")
        tables.mutations.add_row(site=4, node=3, derived_state="T")
        tables.mutations.add_row(site=4, node=3, derived_state="A")
        tables.mutations.add_row(site=4, node=3, derived_state="C")
        tables.mutations.add_row(site=5, node=4, derived_state="C")
        tables.mutations.add_row(site=5, node=4, derived_state="C")
        tables.mutations.add_row(site=5, node=1, derived_state="A")

        ts = tables.tree_sequence()
        model = trait_model.TraitModelAdditive(trait_mean=2, trait_sd=0)
        simulator = simulate_phenotype.PhenotypeSimulator(
            ts, num_causal=6, h2=0, model=model, random_seed=1
        )
        genotypic_effect_data, individual_genetic_array = simulator.sim_genetic_value()

        assert np.array_equal(individual_genetic_array, np.array([4, 16]))
        assert np.array_equal(genotypic_effect_data.site_id, np.arange(6))
        assert np.array_equal(
            genotypic_effect_data.effect_size, np.array([2, 2, 2, 2, 2, 2])
        )
        assert np.array_equal(
            genotypic_effect_data.causal_allele,
            np.array(["T", "T", "C", "T", "C", "C"]),
        )
        assert np.allclose(
            genotypic_effect_data.allele_frequency,
            np.array([0.5, 0.75, 0.25, 0.25, 0.25, 0.5]),
        )
