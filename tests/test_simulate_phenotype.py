import pytest

import msprime
import numpy as np
import tskit
import pandas as pd

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
    @pytest.mark.parametrize("num_ind", [1,2,5])
    @pytest.mark.parametrize("num_causal", [1,2,3])
    @pytest.mark.parametrize("h2", [0.1, 0.5])
    @pytest.mark.parametrize("random_seed", [1,2])
    def test_output_dim_additive(self, num_ind, num_causal, h2, random_seed):
        model = trait_model.TraitModelAdditive(0,1)
        ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, random_seed=random_seed)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(ts, num_causal, model, h2, random_seed)
        
        assert len(phenotype_result.__dict__) == 4
        assert len(genetic_result.__dict__) == 4
        
        assert len(phenotype_result.individual_id) == num_ind
        assert len(phenotype_result.phenotype) == num_ind
        assert len(phenotype_result.environment_noise) == num_ind
        assert len(phenotype_result.genetic_value) == num_ind
        
        assert len(genetic_result.site_id) == num_causal
        assert len(genetic_result.causal_state) == num_causal
        assert len(genetic_result.effect_size) == num_causal
        assert len(genetic_result.allele_frequency) == num_causal

    @pytest.mark.parametrize("num_ind", [1,2,5])
    @pytest.mark.parametrize("num_causal", [1,2,3])
    @pytest.mark.parametrize("h2", [0.1, 0.5])
    @pytest.mark.parametrize("random_seed", [1,2])
    def test_output_dim_Allele(self, num_ind, num_causal, h2, random_seed):
        model = trait_model.TraitModelAllele(0,1)
        ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, random_seed=random_seed)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(ts, num_causal, model, h2, random_seed)
        
        assert len(phenotype_result.__dict__) == 4
        assert len(genetic_result.__dict__) == 4
        
        assert len(phenotype_result.individual_id) == num_ind
        assert len(phenotype_result.phenotype) == num_ind
        assert len(phenotype_result.environment_noise) == num_ind
        assert len(phenotype_result.genetic_value) == num_ind
        
        assert len(genetic_result.site_id) == num_causal
        assert len(genetic_result.causal_state) == num_causal
        assert len(genetic_result.effect_size) == num_causal
        assert len(genetic_result.allele_frequency) == num_causal
        
        # assert max(genetic_result.frequency < 1) and min(genetic_result.frequency > 0)

    @pytest.mark.parametrize("num_ind", [1,2,5])
    @pytest.mark.parametrize("num_causal", [1,2,3])
    @pytest.mark.parametrize("h2", [0.1, 0.5])
    @pytest.mark.parametrize("random_seed", [1,2])
    def test_output_dim_LDAK(self, num_ind, num_causal, h2, random_seed):
        model = trait_model.TraitModelLDAK(0,1, -1)
        ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, random_seed=random_seed)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(ts, num_causal, model, h2, random_seed)
        
        assert len(phenotype_result.__dict__) == 4
        assert len(genetic_result.__dict__) == 4
        
        assert len(phenotype_result.individual_id) == num_ind
        assert len(phenotype_result.phenotype) == num_ind
        assert len(phenotype_result.environment_noise) == num_ind
        assert len(phenotype_result.genetic_value) == num_ind
        
        assert len(genetic_result.site_id) == num_causal
        assert len(genetic_result.causal_state) == num_causal
        assert len(genetic_result.effect_size) == num_causal
        assert len(genetic_result.allele_frequency) == num_causal               

class Test_sim_phenotype_input:
    @pytest.mark.parametrize("ts", [0, "a", [1,1]])
    def test_ts(self, ts):
        model = trait_model.TraitModelAdditive(0,1)
        with pytest.raises(TypeError, match="Input should be a tree sequence data"):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(ts, 2, model, 0.3, 1)

    @pytest.mark.parametrize("num_ind", [1,2,5])
    @pytest.mark.parametrize("num_causal", [1,2,3])
    @pytest.mark.parametrize("h2", [0.1, 0.5])
    @pytest.mark.parametrize("random_seed", [1,2])
    def test_no_mutation(self, num_ind, num_causal, h2, random_seed):
        model = trait_model.TraitModelAdditive(0,1)
        ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, random_seed=random_seed)
        with pytest.raises(ValueError, match="No mutation in the provided data"):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(ts, num_causal, model, h2, random_seed)        

    @pytest.mark.parametrize("model", [None, 1, "a"])
    def test_model(self, model):
        ts = msprime.sim_ancestry(5, sequence_length=100_000, random_seed=2)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=2)
        with pytest.raises(TypeError, match="Mutation model must be an instance of TraitModel"):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(ts, 3, model, 0.3, 2)
    
    @pytest.mark.parametrize("h2", ["0", "a", [1,1]])
    def test_h2(self, h2):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        model = trait_model.TraitModelAdditive(0,1)
        with pytest.raises(TypeError, match="Heritability should be 0 <= h2 <= 1"):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(ts, 2, model, h2, 1)
            
    @pytest.mark.parametrize("h2", [-1, -0.1, 1.01])
    def test_h2_value(self, h2):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        model = trait_model.TraitModelAdditive(0,1)
        with pytest.raises(ValueError, match="Heritability should be 0 <= h2 <= 1"):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(ts, 2, model, h2, 1)            
    
    @pytest.mark.parametrize("num_ind", [1,2,5])
    @pytest.mark.parametrize("num_causal", [1,2,3])
    @pytest.mark.parametrize("random_seed", [1,2])
    def test_h2_zero(self, num_ind, num_causal, random_seed):
        model = trait_model.TraitModelAdditive(0,1)
        ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, random_seed=random_seed)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(ts, num_causal, model, 0, random_seed)
        
        assert np.allclose(phenotype_result.phenotype, phenotype_result.environment_noise)

    @pytest.mark.parametrize("num_ind", [1,2,5])
    @pytest.mark.parametrize("num_causal", [1,2,3])
    @pytest.mark.parametrize("random_seed", [1,2])
    def test_h2_one(self, num_ind, num_causal, random_seed):
        model = trait_model.TraitModelAdditive(0,1)
        ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, random_seed=random_seed)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=random_seed)
        phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(ts, num_causal, model, 1, random_seed)
        
        assert np.allclose(phenotype_result.phenotype, phenotype_result.genetic_value)
        assert np.array_equiv(phenotype_result.environment_noise, np.zeros(num_ind))

    @pytest.mark.parametrize("num_causal", ["1", "a", [1,1]])
    def test_num_causal(self, num_causal):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        model = trait_model.TraitModelAdditive(0,1)
        with pytest.raises(TypeError, match="Number of causal sites should be a positive integer"):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(ts, num_causal, model, 0.3, 1)
            
    @pytest.mark.parametrize("num_causal", [-1, 1.8, -1.5, 0])
    def test_num_causal_value(self, num_causal):
        ts = msprime.sim_ancestry(2, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        model = trait_model.TraitModelAdditive(0,1)
        with pytest.raises(ValueError, match="Number of causal sites should be a positive integer"):
            phenotype_result, genetic_result = simulate_phenotype.sim_phenotype(ts, num_causal, model, 0.3, 1) 

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
        ts = tskit.Tree.generate_comb(4, span=10).tree_sequence
        tables = ts.dump_tables()
        for j in range(9):
            tables.sites.add_row(j, "A")
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
           
        model = trait_model.TraitModelAdditive(0,1)
        rng = np.random.default_rng(1)
        simulateClass = simulate_phenotype.SimPhenotype(ts, 1, 0.3, model, rng)
        tree = ts.first()
        
        g1 = simulateClass._site_genotypes(tree, ts.site(0), "T")
        g2 = simulateClass._site_genotypes(tree, ts.site(1), "T")
        g3 = simulateClass._site_genotypes(tree, ts.site(2), "T")
        g4 = simulateClass._site_genotypes(tree, ts.site(3), "T")
        g5 = simulateClass._site_genotypes(tree, ts.site(4), "T")
        g6 = simulateClass._site_genotypes(tree, ts.site(5), "T")
        g7 = simulateClass._site_genotypes(tree, ts.site(5), "A")
        g8 = simulateClass._site_genotypes(tree, ts.site(6), "T")
        g9 = simulateClass._site_genotypes(tree, ts.site(6), "C")
        g10 = simulateClass._site_genotypes(tree, ts.site(7), "T")
        g11 = simulateClass._site_genotypes(tree, ts.site(8), "T")

        c1 = simulateClass._obtain_allele_frequency(tree, ts.site(0))
        c2 = simulateClass._obtain_allele_frequency(tree, ts.site(1))
        c3 = simulateClass._obtain_allele_frequency(tree, ts.site(2))
        c4 = simulateClass._obtain_allele_frequency(tree, ts.site(3))
        c5 = simulateClass._obtain_allele_frequency(tree, ts.site(4))
        c6 = simulateClass._obtain_allele_frequency(tree, ts.site(5))
        c7 = simulateClass._obtain_allele_frequency(tree, ts.site(6))
        c8 = simulateClass._obtain_allele_frequency(tree, ts.site(7))
        c9 = simulateClass._obtain_allele_frequency(tree, ts.site(8))

        assert np.array_equal(g1, np.array([1,0,0,0]))
        assert np.array_equal(g2, np.array([0,0,1,1]))
        assert np.array_equal(g3, np.array([0,1,0,0]))
        assert np.array_equal(g4, np.array([0,1,1,1]))
        assert np.array_equal(g5, np.array([1,0,1,1]))
        assert np.array_equal(g6, np.array([0,0,0,0]))
        assert np.array_equal(g7, np.array([1,1,1,1]))
        assert np.array_equal(g8, np.array([0,0,0,0]))
        assert np.array_equal(g9, np.array([1,0,0,0]))
        assert np.array_equal(g10, np.array([0,1,0,0]))
        assert np.array_equal(g11, np.array([0,1,1,1]))
        
        assert c1 == {"T": 1}
        assert c2 == {"T": 2}
        assert c3 == {"T": 1}
        assert c4 == {"T": 3}
        assert c5 == {"T": 3}
        assert c6 == {"A": 4}
        assert c7 == {"C": 1}
        assert c8 == {"T": 1, "C": 2}
        assert c9 == {"T": 3}
        
    
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
        tables.mutations.add_row(site=0, node=6, derived_state="T")
        tables.mutations.add_row(site=1, node=0, derived_state="T")
        tables.mutations.add_row(site=2, node=6, derived_state="T")
        tables.mutations.add_row(site=2, node=6, derived_state="C", parent=2)
        tables.mutations.add_row(site=2, node=5, derived_state="T", parent=3)
        
        ts = tables.tree_sequence()
        model = trait_model.TraitModelAdditive(0,1)
        rng = np.random.default_rng(1)
        simulateClass = simulate_phenotype.SimPhenotype(ts, 1, 0.3, model, rng)
        tree = ts.first()
        
        g1 = simulateClass._site_genotypes(tree, ts.site(0), "T")
        g2 = simulateClass._site_genotypes(tree, ts.site(1), "T")
        g3 = simulateClass._site_genotypes(tree, ts.site(2), "C")
        
        c1 = simulateClass._obtain_allele_frequency(tree, ts.site(0))
        c2 = simulateClass._obtain_allele_frequency(tree, ts.site(1))
        c3 = simulateClass._obtain_allele_frequency(tree, ts.site(2))
        
        assert np.array_equal(g1, np.array([0,0,0,1,1,1]))
        assert np.array_equal(g2, np.array([1,0,0,0,0,0]))
        assert np.array_equal(g3, np.array([0,0,0,1,1,0]))
        
        assert c1 == {"T": 3}
        assert c2 == {"T": 1}
        assert c3 == {"C": 2, "T": 1}
        
    
class Test_group_by_individual_genetic_value:
    @pytest.mark.parametrize("num_ind", [1,2,5])
    def test_add(self, num_ind):
        model = trait_model.TraitModelAdditive(0,1)
        ts = msprime.sim_ancestry(num_ind, sequence_length=100_000, random_seed=1)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=1)
        rng = np.random.default_rng(1)
        simulateClass = simulate_phenotype.SimPhenotype(ts, 1, 0.3, model, rng)
        node_value = np.array(range(2 * num_ind))
        individual_genetic = simulateClass._group_by_individual_genetic_value(node_value)
        G = node_value[::2] + node_value[1::2]
        
        assert np.array_equal(G, individual_genetic)

class Test_obtain_allele_frequency:
    def test_binary_tree(self):
        ts = tskit.Tree.generate_comb(6, span=10).tree_sequence
        tables = ts.dump_tables()
        for j in range(4):
            tables.sites.add_row(j, "A")
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
        ts = tables.tree_sequence()
           
        model = trait_model.TraitModelAdditive(0,1)
        rng = np.random.default_rng(1)
        simulateClass = simulate_phenotype.SimPhenotype(ts, 1, 0.3, model, rng)
        tree = ts.first()
        
        g1 = simulateClass._obtain_allele_frequency(tree, ts.site(0))
        g2 = simulateClass._obtain_allele_frequency(tree, ts.site(1))
        g3 = simulateClass._obtain_allele_frequency(tree, ts.site(2))
        g4 = simulateClass._obtain_allele_frequency(tree, ts.site(3))
        
        assert g1 == {"T": 3}
        assert g2 == {"A": 6}
        assert g3 == {"T": 4, "C": 1}
        assert g4 == {"G": 2, "C": 1}
        
        


"""
class Test_update_node_values_array_access:
    def test_update(self):
        # Construct an artificial tree with 6 leaves with node 7 and 9 missing from the tree
        # We will be having mutation in every edge having genetic value 1 to be sure that all the edges are captured in the tree traversal algorithm
        root = 12
        left_child_array = np.array([-1,-1,-1,-1,-1,-1,0,-1,2,-1,4,6,11,12])
        right_sib_array = np.array([1,-1,3,-1,5,-1,8,-1,-1,-1,-1,10,-1,-1])
        node_values = np.array([1,1,1,1,1,1,1,0,1,0,1,1,1])
        
        # Create an arbitrary genotype vector
        rng = np.random.default_rng(1)
        G = rng.normal(loc=0, scale=1, size=6)        
        G1 = np.copy(G)
        sim_pheno.update_node_values_array_access(root, left_child_array, right_sib_array, node_values, G)
        
        actual_G = np.array([4,4,4,4,3,3]) + G1
        
        assert np.array_equal(G, actual_G)
      
    def binary_tree(self):
        # 3.00   6     
        #     ┊ ┏━┻━┓    ┊
        # 2.00┊ ┃   5   ┊
        #     ┊ ┃ ┏━┻┓   ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓  ┊
        # 0.00 0 1 2 3 
        #              
        # Mutation in all edges
        ts = tskit.Tree.generate_comb(4, span=10).tree_sequence
        tables = ts.dump_tables()
        for j in range(6):
            tables.sites.add_row(j, "A")
            tables.mutations.add_row(site=j, node=j, derived_state="T")
        ts = tables.tree_sequence()
        tree = ts.first()
        node_values = np.zeros(ts.num_nodes)
        for mut in tree.mutations():
            node_values[mut.node] += 1
        
        rng = np.random.default_rng(2)
        G = rng.normal(loc=0, scale=1, size=ts.num_samples)
        G1 = np.copy(G)
        sim_pheno.update_node_values_array_access(tree.root, tree.left_child_array, tree.right_sib_array, node_values, G)
        
        actual_G = np.array([1,2,3,3]) + G1
        
        assert np.array_equal(G, actual_G)
    
    def non_binary_tree(self):

        # 2.00      7   
        #     ┊ ┏━┏━━┏━┻━━━┓   ┊
        # 1.00┊ ┃ ┃ ┃    6   ┊
        #     ┊ ┃ ┃ ┃  ┏━┳┻┳━┓ ┊
        # 0.00 0 1 2  3 4 5
        #              
        # Mutation in all edges
        ts = tskit.Tree.generate_balanced(6, arity=4, span=10).tree_sequence
        tables = ts.dump_tables()
        for j in range(7):
            tables.sites.add_row(j, "A")
            tables.mutations.add_row(site=j, node=j, derived_state="T")
        ts = tables.tree_sequence()
        tree = ts.first()
        node_values = np.zeros(ts.num_nodes)
        for mut in tree.mutations():
            node_values[mut.node] += 1
        
        rng = np.random.default_rng(3)
        G = rng.normal(loc=0, scale=1, size=ts.num_samples)
        G1 = np.copy(G)
        sim_pheno.update_node_values_array_access(tree.root, tree.left_child_array, tree.right_sib_array, node_values, G)
        
        actual_G = np.array([1,1,1,2,2,2]) + G1
        
        assert np.array_equal(G, actual_G)

class Test_environment:
    @pytest.mark.parametrize("size", [1, 2, 10, 100])
    @pytest.mark.parametrize("h2", [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
    def test_environment(self, size, h2):
        rng = np.random.default_rng(1)
        G = rng.normal(size = size)
        phenotype, E = sim_pheno.environment(G, h2, 1, rng)
        assert len(phenotype) == size
        assert len(E) == size
        assert np.issubdtype(phenotype.dtype, np.floating)
        assert np.issubdtype(E.dtype, np.floating)
    
    @pytest.mark.parametrize("h2", [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
    def test_environment_G_error(self, h2):
        rng = np.random.default_rng(2)
        G = rng.normal(size = 0)
        with pytest.raises(ValueError, match="No individuals in the simulation model"):
            phenotype, E = sim_pheno.environment(G, h2, 1, rng)
    
    @pytest.mark.parametrize("size", [1, 2, 10, 100])    
    @pytest.mark.parametrize("h2", [-0.2, -0.01, 1.01, 1.2, "a", "1", [0.4,0.3]])
    def test_h2error(self, size, h2):
        G = np.random.normal(size = size)
        rng = np.random.default_rng(3)        
        with pytest.raises(ValueError, match="Heritability should be 0 <= h2 <= 1"):
            phenotype, E = sim_pheno.environment(G, h2, 1, rng)   

class Test_choose_causal:
    @pytest.mark.parametrize("num_mutations", [10, 100, 1000])
    @pytest.mark.parametrize("num_causal", [1, 2, 10])
    @pytest.mark.parametrize("trait_sd", [0.1, 0.3, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize("trait_mean", [-1,0,1])
    def test_choose_causal(self, num_mutations, num_causal, trait_sd, trait_mean):
        rng = np.random.default_rng(1)
        mutation_id, beta = sim_pheno.choose_causal(num_mutations, num_causal, trait_mean, trait_sd, rng)
        assert len(beta) == num_causal
        assert len(mutation_id) == num_causal
        assert min(mutation_id) >= 0
        assert max(mutation_id) < num_mutations
        assert np.issubdtype(beta.dtype, np.floating)
        assert np.issubdtype(mutation_id.dtype, np.integer)

    @pytest.mark.parametrize("num_mutations", [10, 100, 1000])
    @pytest.mark.parametrize("trait_sd", [0.1, 0.3, 0.5, 0.7, 0.9])   
    @pytest.mark.parametrize("trait_mean", [-1,0,1])    
    def test_zero(self, num_mutations, trait_sd, trait_mean):
        rng = np.random.default_rng(1)
        num_causal = 0
        mutation_id, beta = sim_pheno.choose_causal(num_mutations, num_causal, trait_mean, trait_sd, rng)
        assert len(beta) == 0
        assert len(mutation_id) == 0
    
    @pytest.mark.parametrize("addition", [1, 2, 10, 100])
    @pytest.mark.parametrize("num_mutations", [1, 2, 10, 100])
    @pytest.mark.parametrize("trait_sd", [0.1, 0.3, 0.5, 0.7, 0.9]) 
    @pytest.mark.parametrize("trait_mean", [-1,0,1])
    def test_error_num_sites(self, addition, num_mutations, trait_sd, trait_mean):
        rng = np.random.default_rng(2)
        with pytest.raises(ValueError, match="There are more causal sites than the number of mutations inside the tree sequence"):
            num_causal = num_mutations + addition
            mutation_id, beta = sim_pheno.choose_causal(num_mutations, num_causal, trait_mean, trait_sd, rng)
    
    @pytest.mark.parametrize("num_mutations", [10, 100, 1000])
    @pytest.mark.parametrize("trait_sd", [0.1, 0.3, 0.5, 0.7, 0.9]) 
    @pytest.mark.parametrize("num_causal", [-1, 0]) 
    @pytest.mark.parametrize("trait_mean", [-1,0,1])
    def test_error_num_causal(self, num_mutations, trait_sd, num_causal, trait_mean):
        rng = np.random.default_rng(3)
        with pytest.raises(ValueError, match = "Number of causal sites should be a non-negative integer"):
            mutation_id, beta = sim_pheno.choose_causal(num_mutations, num_causal, trait_mean, trait_sd, rng)
        
    @pytest.mark.parametrize("num_mutations", [10, 100, 1000])
    @pytest.mark.parametrize("trait_sd", [0, -0.1, -1, -10])
    @pytest.mark.parametrize("num_causal", [1, 2, 10]) 
    @pytest.mark.parametrize("trait_mean", [-1,0,1])    
    def test_error_trait_sd(self, num_mutations, trait_sd, num_causal, trait_mean):
        rng = np.random.default_rng(np.random.randint(5))
        with pytest.raises(ValueError, match = "Standard deviation should be a non-negative number"):
            mutation_id, beta = sim_pheno.choose_causal(num_mutations, num_causal, trait_mean, trait_sd, rng)



class Test_genetic_value:
    @pytest.mark.parametrize("seed", [1, 3, 5, 7, 9])
    @pytest.mark.parametrize("num_causal", [1, 2, 10])
    def test_genetic_value(self, seed, num_causal):
        rng = np.random.default_rng(seed)
        ts = msprime.sim_ancestry(100, sequence_length=10_000, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
        mutation_id, beta = sim_pheno.choose_causal(ts.num_mutations, num_causal,0, 1, rng)
        G, location, mutation_list = sim_pheno.genetic_value(ts, mutation_id, beta)
        assert len(G) == ts.num_individuals
        assert len(location) == num_causal
        assert len(mutation_list) == num_causal
        assert np.issubdtype(G.dtype, np.floating)
        assert np.issubdtype(location.dtype, np.integer) or np.issubdtype(location.dtype, np.floating)
        assert np.issubdtype(mutation_list.dtype, np.integer)
    
    def test_tree_sequence_binary(self):
        ts = tskit.Tree.generate_comb(6, span=20).tree_sequence
        tables = ts.dump_tables()
        for j in range(10):
            tables.sites.add_row(j, "A")
            tables.mutations.add_row(site=j, node=j, derived_state="T")
        ts = tables.tree_sequence()
        mutation_id = np.array(list(range(10)))
        beta = np.array(list(range(10)))
        G = sim_pheno.genetic_value(ts, mutation_id, beta)
        
        G = G[::2] + G[1::2]
        
        G_actual = np.array([10, 46, 69])

        assert np.array_equal(G, G_actual)
        
    def test_tree_sequence_nonbinary(self):
        rng = np.random.default_rng(2)
        ts = tskit.Tree.generate_balanced(6, arity=4, span=10).tree_sequence
        tables = ts.dump_tables()
        for j in range(7):
            tables.sites.add_row(j, "A")
            tables.mutations.add_row(site=j, node=j, derived_state="T")
        ts = tables.tree_sequence()
        mutation_id = np.array([0, 1, 3, 6])
        beta = np.repeat(1,4)
        G = sim_pheno.genetic_value(ts, mutation_id, beta)
        G = G[::2] + G[1::2]
        G_actual = np.array([2,2,2])
        
        assert np.array_equal(G, G_actual)
              
class Test_phenotype_sim:
    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    @pytest.mark.parametrize("num_causal", [1, 2, 10])
    def test_phenotype_sim(self, seed, num_causal):
        size = 20
        ts = msprime.sim_ancestry(size, sequence_length=100_000, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed, model="binary", discrete_genome=False)
        pheno_df, gene_df = sim_pheno.phenotype_sim(ts, num_causal, trait_sd=1, h2=0.3, seed=seed)
        assert pheno_df.isnull().values.any() == False
        assert gene_df.isnull().values.any() == False
        assert len(pheno_df) == ts.num_individuals
        assert len(gene_df) == num_causal
    
    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    @pytest.mark.parametrize("num_causal", [1, 2, 10])
    def test_no_mutation(self, seed, num_causal):
        size = 20
        rng = np.random.default_rng(seed)
        ts = msprime.sim_ancestry(size, sequence_length=100_000, recombination_rate=1e-8,population_size=10**4, random_seed=seed)
        with pytest.raises(ValueError, match = "No mutation in the provided data"):
            pheno_df, gene_df = sim_pheno.phenotype_sim(ts, num_causal, trait_sd=1, h2=0.3, seed=seed)
"""