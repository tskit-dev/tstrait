import msprime
import numpy as np
import pandas as pd
import pytest
import tskit

import tstrait
from tstrait.base import _check_numeric_array

from .data import (
    all_trees_ts,
    allele_freq_one,
    binary_tree,
    binary_tree_seq,
    diff_ind_tree,
    non_binary_tree,
    simple_tree_seq,
    triploid_tree,
)  # noreorder

# from tstrait.trait import sim_trait


@pytest.fixture(scope="class")
def sample_ts():
    ts = msprime.sim_ancestry(10, sequence_length=100_000, random_seed=1)
    ts = msprime.sim_mutations(ts, rate=1e-2, random_seed=1)
    return ts


@pytest.fixture(scope="class")
def sample_df():
    df = pd.DataFrame(
        {
            "site_id": [0, 1],
            "causal_allele": ["A", "A"],
            "effect_size": [0.1, 0.1],
            "trait_id": [0, 0],
            "allele_freq": [0.2, 0.3],
        }
    )

    return df


@pytest.fixture(scope="class")
def sample_trait_model():
    return tstrait.trait_model(distribution="normal", mean=0, var=1)


def individual_genetic_values(tree, site, causal_allele, effect_size):
    """Return individual values for one causal site via the public API."""
    trait_df = pd.DataFrame(
        {
            "site_id": [site.id],
            "effect_size": [effect_size],
            "trait_id": [0],
            "causal_allele": [causal_allele],
        }
    )
    result = tstrait.genetic_value(ts=tree.tree_sequence, trait_df=trait_df)
    return result["genetic_value"].to_numpy()


def naive_edge_effect(ts, trait_df):
    """Reference implementation of edge_effect, computed from allele states.

    Rather than working from the mutations directly, this determines the allele
    that each node carries at a causal site and assigns to every edge the change
    in causal allele state between its parent and child node.
    """
    num_trait = int(np.max(trait_df["trait_id"])) + 1
    edge_effect_table = np.zeros((num_trait, ts.num_edges))
    for data in trait_df.itertuples():
        site = ts.site(data.site_id)
        tree = ts.at(site.position)
        state = {u: site.ancestral_state for u in tree.nodes()}
        for m in site.mutations:
            if tree.parent(m.node) == tskit.NULL:
                if (m.derived_state == data.causal_allele) != (
                    m.inherited_state == data.causal_allele
                ):
                    raise ValueError(
                        "Cannot assign an edge effect to a mutation on a root node"
                    )
            for u in tree.nodes(m.node):
                state[u] = m.derived_state
        for u in state:
            edge = tree.edge(u)
            if edge != tskit.NULL:
                state_change = int(state[u] == data.causal_allele) - int(
                    state[tree.parent(u)] == data.causal_allele
                )
                edge_effect_table[data.trait_id, edge] += state_change * data.effect_size
    return pd.DataFrame(
        {
            "trait_id": np.repeat(np.arange(num_trait), ts.num_edges),
            "edge_id": np.tile(np.arange(ts.num_edges), num_trait),
            "effect_size": edge_effect_table.flatten(),
        }
    )


def sites_without_root_mutations(ts):
    """Return the IDs of the sites where no mutation sits above a local root."""
    site_id = [
        site.id
        for site in ts.sites()
        if all(ts.at(site.position).parent(m.node) != tskit.NULL for m in site.mutations)
    ]
    return np.array(site_id, dtype=int)


def random_trait_df(ts, num_trait, seed, site_id=None):
    """Return a trait dataframe over a sample of the sites in ``ts``.

    The causal allele of a row is drawn from the alleles that occur at its site,
    so that the effects are not trivially zero.
    """
    rng = np.random.default_rng(seed)
    if site_id is None:
        site_id = np.arange(ts.num_sites)
    num_site = min(len(site_id), 10)
    assert num_site > 0
    site_id = np.sort(rng.choice(site_id, size=num_site, replace=False))
    causal_allele = []
    for site in site_id:
        alleles = [str(ts.sites_ancestral_state[site])]
        alleles.extend(map(str, ts.mutations_derived_state[ts.mutations_site == site]))
        causal_allele.extend(rng.choice(alleles, size=num_trait))
    return pd.DataFrame(
        {
            "site_id": np.repeat(site_id, num_trait),
            "effect_size": rng.normal(size=num_site * num_trait),
            "trait_id": np.tile(np.arange(num_trait), num_site),
            "causal_allele": causal_allele,
        }
    )


def mutated_binary_tree(sites):
    """Return binary_tree() with its sites and mutations replaced.

    ``sites`` is a list of ``(position, ancestral_state, mutations)`` tuples,
    where each mutation is a ``(node, derived_state)`` pair. Repeated mutations
    on a node are chained together through the mutation parent.
    """
    tables = binary_tree().dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    for position, ancestral_state, mutations in sites:
        site_id = tables.sites.add_row(
            position=position, ancestral_state=ancestral_state
        )
        parent = {}
        for node, derived_state in mutations:
            parent[node] = tables.mutations.add_row(
                site=site_id,
                node=node,
                derived_state=derived_state,
                parent=parent.get(node, tskit.NULL),
            )
    return tables.tree_sequence()


def multi_allelic_mutations(ts, rate, seed):
    """Add mutations under a model that gives recurrent and back mutations.

    The genome is discrete, so sites generally carry several mutations.
    """
    return msprime.sim_mutations(ts, rate=rate, model=msprime.JC69(), random_seed=seed)


class TestInput:
    """This test will check that an informative error is raised when the input parameter
    does not have an appropriate type or value.
    """

    @pytest.mark.parametrize("function", [tstrait.genetic_value, tstrait.edge_effect])
    def test_input_type(self, sample_ts, sample_df, function):
        with pytest.raises(
            TypeError, match="ts must be a <class 'tskit.trees.TreeSequence'> instance"
        ):
            function(ts=1, trait_df=sample_df)
        with pytest.raises(
            TypeError,
            match=f"trait_df must be a {pd.DataFrame} instance",
        ):
            function(ts=sample_ts, trait_df=1)

    @pytest.mark.parametrize("function", [tstrait.genetic_value, tstrait.edge_effect])
    @pytest.mark.parametrize("column", ["site_id", "effect_size", "trait_id"])
    def test_missing_column(self, sample_ts, sample_df, function, column):
        with pytest.raises(
            ValueError, match="columns must be included in trait_df dataframe"
        ):
            function(ts=sample_ts, trait_df=sample_df.drop(columns=[column]))

    @pytest.mark.parametrize("function", [tstrait.genetic_value, tstrait.edge_effect])
    def test_bad_input(self, sample_ts, sample_df, function):
        with pytest.raises(ValueError, match="site_id must be non-decreasing"):
            df = sample_df.copy()
            df["site_id"] = [2, 0]
            function(ts=sample_ts, trait_df=df)

    @pytest.mark.parametrize("function", [tstrait.genetic_value, tstrait.edge_effect])
    @pytest.mark.parametrize("site_id", [[-2, -1], [0, 10**6]])
    def test_site_id_out_of_bounds(self, sample_ts, sample_df, function, site_id):
        with pytest.raises(
            ValueError, match=f"site_id must be in \\[0, {sample_ts.num_sites}\\)"
        ):
            df = sample_df.copy()
            df["site_id"] = site_id
            function(ts=sample_ts, trait_df=df)

    @pytest.mark.parametrize("function", [tstrait.genetic_value, tstrait.edge_effect])
    @pytest.mark.parametrize("trait_id", [[2, 3], [0, 2]])
    def test_trait_id(self, sample_ts, sample_df, function, trait_id):
        with pytest.raises(
            ValueError, match="trait_id must be consecutive and start from 0"
        ):
            df = sample_df.copy()
            df["trait_id"] = trait_id
            function(ts=sample_ts, trait_df=df)

    def test_no_individual(self, sample_df):
        ts = msprime.simulate(10, length=100, random_seed=1)
        # legacy msprime.simulate() creates a tree sequence without individuals
        with pytest.raises(
            ValueError, match="No individuals in the provided tree sequence dataset"
        ):
            tstrait.genetic_value(ts=ts, trait_df=sample_df)

    @pytest.mark.parametrize(
        ("level", "id_column", "size_attribute"),
        [
            ("node", "node_id", "num_nodes"),
            ("edge", "edge_id", "num_edges"),
        ],
    )
    def test_node_or_edge_level_without_individuals(
        self, sample_df, level, id_column, size_attribute
    ):
        ts = msprime.simulate(10, length=100, mutation_rate=1e-2, random_seed=1)
        result = tstrait.genetic_value(ts=ts, trait_df=sample_df, level=level)

        size = getattr(ts, size_attribute)
        assert list(result.columns) == ["trait_id", id_column, "genetic_value"]
        assert len(result) == size
        np.testing.assert_array_equal(result["trait_id"], np.zeros(size))
        np.testing.assert_array_equal(result[id_column], np.arange(size))

    def test_invalid_level(self, sample_ts, sample_df):
        with pytest.raises(
            ValueError,
            match="level must be one of 'individual', 'node', or 'edge'",
        ):
            tstrait.genetic_value(ts=sample_ts, trait_df=sample_df, level="bla")

    def test_empty_trait_df(self, sample_ts, sample_df):
        empty_trait_df = sample_df.iloc[0:0]
        with pytest.raises(ValueError, match="trait_df must contain at least one row"):
            tstrait.genetic_value(ts=sample_ts, trait_df=empty_trait_df)
        with pytest.raises(ValueError, match="trait_df must contain at least one row"):
            tstrait.edge_effect(ts=sample_ts, trait_df=empty_trait_df)


class TestEdgeEffect:
    @pytest.mark.parametrize(
        ("ancestral_state", "derived_state", "expected_effect"),
        [
            pytest.param("A", "T", 2, id="non_causal_to_causal"),
            pytest.param("T", "A", -2, id="causal_to_non_causal"),
            pytest.param("A", "G", 0, id="non_causal_to_non_causal"),
            pytest.param("T", "T", 0, id="causal_to_causal"),
        ],
    )
    def test_causal_allele_state_transition(
        self, ancestral_state, derived_state, expected_effect
    ):
        ts = binary_tree()
        tables = ts.dump_tables()
        tables.mutations.clear()
        tables.sites.clear()
        site_id = tables.sites.add_row(position=0, ancestral_state=ancestral_state)
        mutation_node = 4
        tables.mutations.add_row(
            site=site_id, node=mutation_node, derived_state=derived_state
        )
        ts = tables.tree_sequence()

        trait_df = pd.DataFrame(
            {
                "site_id": [site_id],
                "effect_size": [2],
                "trait_id": [0],
                "causal_allele": ["T"],
            }
        )
        result = tstrait.edge_effect(ts=ts, trait_df=trait_df)

        expected = np.zeros(ts.num_edges)
        expected[ts.first().edge(mutation_node)] = expected_effect
        np.testing.assert_array_equal(result["effect_size"], expected)

    def test_site_mode_allelic_state_transitions(self):
        ts = binary_tree()
        tables = ts.dump_tables()
        tables.mutations.clear()
        tables.sites.clear()
        site_id = tables.sites.add_row(position=0, ancestral_state="A")
        mutation_id = tables.mutations.add_row(site=site_id, node=4, derived_state="T")
        tables.mutations.add_row(
            site=site_id, node=0, derived_state="A", parent=mutation_id
        )
        tables.mutations.add_row(site=site_id, node=3, derived_state="T")
        ts = tables.tree_sequence()

        trait_df = pd.DataFrame(
            {
                "site_id": [site_id],
                "effect_size": [2],
                "trait_id": [0],
                "causal_allele": ["T"],
            }
        )
        result = tstrait.edge_effect(ts=ts, trait_df=trait_df)

        tree = ts.first()
        expected = np.zeros(ts.num_edges)
        expected[tree.edge(4)] = 2
        expected[tree.edge(0)] = -2
        expected[tree.edge(3)] = 2
        np.testing.assert_array_equal(result["effect_size"], expected)

    def test_mutation_on_root(self):
        ts = binary_tree()
        root = ts.first().root
        tables = ts.dump_tables()
        tables.mutations.clear()
        tables.sites.clear()
        site_id = tables.sites.add_row(position=0, ancestral_state="A")
        tables.mutations.add_row(site=site_id, node=root, derived_state="T")
        ts = tables.tree_sequence()
        trait_df = pd.DataFrame(
            {
                "site_id": [site_id],
                "effect_size": [1],
                "trait_id": [0],
                "causal_allele": ["T"],
            }
        )

        with pytest.raises(
            ValueError,
            match="Cannot assign an edge effect to a mutation on a root node",
        ):
            tstrait.edge_effect(ts=ts, trait_df=trait_df)

    def test_output_dim(self):
        ts = mutated_binary_tree([(0, "A", [(4, "T")]), (5, "A", [(5, "T")])])
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 0, 1, 1],
                "effect_size": [1, 2, 3, 4],
                "trait_id": [0, 1, 0, 1],
                "causal_allele": ["T", "T", "T", "T"],
            }
        )
        result = tstrait.edge_effect(ts=ts, trait_df=trait_df)
        assert list(result.columns) == ["trait_id", "edge_id", "effect_size"]
        assert len(result) == 2 * ts.num_edges
        np.testing.assert_array_equal(
            result["trait_id"], np.repeat([0, 1], ts.num_edges)
        )
        np.testing.assert_array_equal(
            result["edge_id"], np.tile(np.arange(ts.num_edges), 2)
        )

    def test_multiple_traits(self):
        """Traits with different causal alleles and effect sizes are kept apart."""
        ts = mutated_binary_tree([(0, "A", [(4, "T")]), (5, "A", [(5, "G")])])
        tree = ts.first()
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 0, 1, 1],
                "effect_size": [1.5, -2.5, 3.5, 4.5],
                "trait_id": [0, 1, 0, 1],
                "causal_allele": ["T", "A", "G", "T"],
            }
        )
        result = tstrait.edge_effect(ts=ts, trait_df=trait_df)

        expected = np.zeros((2, ts.num_edges))
        # Trait 0: site 0 gains "T" on node 4, site 1 gains "G" on node 5.
        expected[0, tree.edge(4)] = 1.5
        expected[0, tree.edge(5)] = 3.5
        # Trait 1: site 0 loses the ancestral "A" on node 4, site 1 is unchanged
        # because "T" occurs at neither end of the "A" -> "G" transition.
        expected[1, tree.edge(4)] = 2.5
        np.testing.assert_array_almost_equal(result["effect_size"], expected.flatten())

    def test_site_shared_between_traits(self):
        ts = mutated_binary_tree([(0, "A", [(4, "T")])])
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 0, 0],
                "effect_size": [1, 2, 4],
                "trait_id": [0, 1, 2],
                "causal_allele": ["T", "T", "T"],
            }
        )
        result = tstrait.edge_effect(ts=ts, trait_df=trait_df)
        expected = np.zeros((3, ts.num_edges))
        expected[:, ts.first().edge(4)] = [1, 2, 4]
        np.testing.assert_array_equal(result["effect_size"], expected.flatten())

    def test_site_without_mutations(self):
        ts = mutated_binary_tree([(0, "A", []), (5, "A", [(4, "T")])])
        trait_df = pd.DataFrame(
            {
                "site_id": [0],
                "effect_size": [2],
                "trait_id": [0],
                "causal_allele": ["T"],
            }
        )
        result = tstrait.edge_effect(ts=ts, trait_df=trait_df)
        np.testing.assert_array_equal(result["effect_size"], np.zeros(ts.num_edges))

    def test_stacked_mutations(self):
        """Mutations chained on one branch telescope to the net state change."""
        ts = mutated_binary_tree([(0, "A", [(4, "T"), (4, "G"), (4, "T")])])
        trait_df = pd.DataFrame(
            {
                "site_id": [0],
                "effect_size": [2],
                "trait_id": [0],
                "causal_allele": ["T"],
            }
        )
        result = tstrait.edge_effect(ts=ts, trait_df=trait_df)
        # The individual transitions contribute +2, -2 and +2.
        expected = np.zeros(ts.num_edges)
        expected[ts.first().edge(4)] = 2
        np.testing.assert_array_equal(result["effect_size"], expected)

    def test_mutations_accumulate_on_one_edge(self):
        """Mutations at different sites on the same branch are summed."""
        ts = mutated_binary_tree([(0, "A", [(4, "T")]), (5, "A", [(4, "T")])])
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 1],
                "effect_size": [1.5, 2.5],
                "trait_id": [0, 0],
                "causal_allele": ["T", "T"],
            }
        )
        result = tstrait.edge_effect(ts=ts, trait_df=trait_df)
        expected = np.zeros(ts.num_edges)
        expected[ts.first().edge(4)] = 4
        np.testing.assert_array_almost_equal(result["effect_size"], expected)

    @pytest.mark.parametrize("effect_size", [-2.5, 0, 0.25])
    def test_effect_size(self, effect_size):
        ts = mutated_binary_tree([(0, "A", [(4, "T")])])
        trait_df = pd.DataFrame(
            {
                "site_id": [0],
                "effect_size": [effect_size],
                "trait_id": [0],
                "causal_allele": ["T"],
            }
        )
        result = tstrait.edge_effect(ts=ts, trait_df=trait_df)
        expected = np.zeros(ts.num_edges)
        expected[ts.first().edge(4)] = effect_size
        np.testing.assert_array_almost_equal(result["effect_size"], expected)

    @pytest.mark.parametrize(
        ("causal_allele", "expected_effect"), [("GGGT", 2), ("AAC", -2), ("G", 0)]
    )
    def test_multi_character_alleles(self, causal_allele, expected_effect):
        ts = mutated_binary_tree([(0, "AAC", [(4, "GGGT")])])
        trait_df = pd.DataFrame(
            {
                "site_id": [0],
                "effect_size": [2],
                "trait_id": [0],
                "causal_allele": [causal_allele],
            }
        )
        result = tstrait.edge_effect(ts=ts, trait_df=trait_df)
        expected = np.zeros(ts.num_edges)
        expected[ts.first().edge(4)] = expected_effect
        np.testing.assert_array_equal(result["effect_size"], expected)

    @pytest.mark.parametrize("is_sample", [True, False])
    def test_mutation_on_isolated_node(self, is_sample):
        """An isolated node has no ancestral edge, just like a root."""
        ts = binary_tree()
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        flags = tskit.NODE_IS_SAMPLE if is_sample else 0
        node = tables.nodes.add_row(flags=flags, time=0)
        site_id = tables.sites.add_row(position=0, ancestral_state="A")
        tables.mutations.add_row(site=site_id, node=node, derived_state="T")
        ts = tables.tree_sequence()
        trait_df = pd.DataFrame(
            {
                "site_id": [site_id],
                "effect_size": [1],
                "trait_id": [0],
                "causal_allele": ["T"],
            }
        )
        with pytest.raises(
            ValueError,
            match="Cannot assign an edge effect to a mutation on a root node",
        ):
            tstrait.edge_effect(ts=ts, trait_df=trait_df)

    def test_mutation_on_root_without_state_change(self):
        """A mutation above a root is fine if it does not change the state."""
        ts = mutated_binary_tree([(0, "A", [(6, "T")])])
        trait_df = pd.DataFrame(
            {
                "site_id": [0],
                "effect_size": [1],
                "trait_id": [0],
                "causal_allele": ["G"],
            }
        )
        result = tstrait.edge_effect(ts=ts, trait_df=trait_df)
        np.testing.assert_array_equal(result["effect_size"], np.zeros(ts.num_edges))

    def test_multiple_roots(self):
        """A mutation above one root of a forest has no edge to sit on."""
        ts = binary_tree()
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        # Detach node 4 and node 5 from the root, leaving two roots.
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 6:
                tables.edges.append(edge)
        site_id = tables.sites.add_row(position=0, ancestral_state="A")
        tables.mutations.add_row(site=site_id, node=4, derived_state="T")
        tables.sort()
        ts = tables.tree_sequence()
        trait_df = pd.DataFrame(
            {
                "site_id": [site_id],
                "effect_size": [1],
                "trait_id": [0],
                "causal_allele": ["T"],
            }
        )
        with pytest.raises(
            ValueError,
            match="Cannot assign an edge effect to a mutation on a root node",
        ):
            tstrait.edge_effect(ts=ts, trait_df=trait_df)

    def test_no_edges(self):
        tables = tskit.TableCollection(sequence_length=10)
        for _ in range(2):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        site_id = tables.sites.add_row(position=0, ancestral_state="A")
        tables.mutations.add_row(site=site_id, node=0, derived_state="T")
        ts = tables.tree_sequence()
        trait_df = pd.DataFrame(
            {
                "site_id": [site_id],
                "effect_size": [1],
                "trait_id": [0],
                "causal_allele": ["G"],
            }
        )
        result = tstrait.edge_effect(ts=ts, trait_df=trait_df)
        assert len(result) == 0


class TestEdgeEffectReference:
    """Compare edge_effect against a state-based reference implementation over
    a range of tree topologies and allele configurations.
    """

    def verify(self, ts, num_trait, seed=1):
        trait_df = random_trait_df(ts, num_trait, seed)
        try:
            expected = naive_edge_effect(ts, trait_df)
        except ValueError:
            with pytest.raises(
                ValueError, match="Cannot assign an edge effect to a mutation"
            ):
                tstrait.edge_effect(ts=ts, trait_df=trait_df)
        else:
            result = tstrait.edge_effect(ts=ts, trait_df=trait_df)
            pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "ts_func",
        [
            binary_tree,
            diff_ind_tree,
            non_binary_tree,
            triploid_tree,
            binary_tree_seq,
            simple_tree_seq,
            allele_freq_one,
        ],
    )
    @pytest.mark.parametrize("num_trait", [1, 3])
    def test_data_tree_sequence(self, ts_func, num_trait):
        self.verify(ts_func(), num_trait)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    @pytest.mark.parametrize("rate", [2.0, 10.0])
    def test_all_trees(self, n, rate):
        ts = multi_allelic_mutations(all_trees_ts(n), rate=rate, seed=n)
        self.verify(ts, num_trait=2)

    @pytest.mark.parametrize("recombination_rate", [0, 1e-7])
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_simulated(self, recombination_rate, seed):
        ts = msprime.sim_ancestry(
            8,
            sequence_length=10_000,
            recombination_rate=recombination_rate,
            population_size=1000,
            random_seed=seed,
        )
        ts = multi_allelic_mutations(ts, rate=1e-4, seed=seed)
        self.verify(ts, num_trait=3, seed=seed)


class TestEdgeEffectAndNodeValues:
    """In a tree sequence with a single tree every edge spans every site, so
    the effect introduced on an edge is exactly the difference between the
    genetic values of its child and parent node.
    """

    def verify(self, ts, num_trait=2, seed=1):
        assert ts.num_trees == 1
        trait_df = random_trait_df(
            ts, num_trait, seed, site_id=sites_without_root_mutations(ts)
        )
        effect_df = tstrait.edge_effect(ts=ts, trait_df=trait_df)
        value_df = tstrait.genetic_value(ts=ts, trait_df=trait_df, level="node")
        for trait_id in range(num_trait):
            effect = effect_df[effect_df["trait_id"] == trait_id]
            value = value_df[value_df["trait_id"] == trait_id]
            value = value.set_index("node_id")["genetic_value"]
            expected = (
                value[ts.edges_child].to_numpy() - value[ts.edges_parent].to_numpy()
            )
            np.testing.assert_allclose(
                effect["effect_size"].to_numpy(), expected, atol=1e-9
            )

    @pytest.mark.parametrize(
        "ts_func", [binary_tree, non_binary_tree, triploid_tree, allele_freq_one]
    )
    def test_data_tree_sequence(self, ts_func):
        self.verify(ts_func())

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_simulated(self, seed):
        ts = msprime.sim_ancestry(
            8, sequence_length=1000, population_size=1000, random_seed=seed
        )
        ts = multi_allelic_mutations(ts, rate=1e-3, seed=seed)
        self.verify(ts, num_trait=3, seed=seed)


class TestOutputDim:
    """Check that the genetic_value function gives the output with correct dimensions"""

    def check_dimensions(self, genetic_df, nrow):
        assert len(genetic_df) == nrow
        assert genetic_df.shape[1] == 3
        assert list(genetic_df.columns) == [
            "trait_id",
            "individual_id",
            "genetic_value",
        ]

        _check_numeric_array(genetic_df["individual_id"], "individual_id")
        _check_numeric_array(genetic_df["genetic_value"], "genetic_value")
        _check_numeric_array(genetic_df["trait_id"], "trait_id")

    def test_output_simple(self, sample_ts, sample_df):
        genetic_result = tstrait.genetic_value(ts=sample_ts, trait_df=sample_df)
        self.check_dimensions(genetic_result, sample_ts.num_individuals)

    @pytest.mark.parametrize(
        "trait_model",
        [
            tstrait.trait_model(distribution="normal", mean=0, var=1),
            tstrait.trait_model(distribution="exponential", scale=1),
            tstrait.trait_model(distribution="fixed", value=1),
            tstrait.trait_model(distribution="t", mean=0, var=1, df=1),
            tstrait.trait_model(distribution="gamma", shape=1, scale=1),
        ],
    )
    def test_output_sim_trait(self, sample_ts, trait_model):
        num_causal = 10
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=num_causal, model=trait_model, random_seed=10
        )
        genetic_result = tstrait.genetic_value(ts=sample_ts, trait_df=trait_df)
        self.check_dimensions(genetic_result, sample_ts.num_individuals)
        np.testing.assert_equal(
            genetic_result["trait_id"], np.zeros(sample_ts.num_individuals)
        )
        np.testing.assert_equal(
            genetic_result["individual_id"],
            np.arange(sample_ts.num_individuals),
        )

    def test_output_multivariate(self, sample_ts):
        num_trait = 3
        mean = np.ones(num_trait)
        cov = np.eye(num_trait)
        num_causal = 10
        model = tstrait.trait_model(distribution="multi_normal", mean=mean, cov=cov)
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=num_causal, model=model, random_seed=5
        )
        genetic_result = tstrait.genetic_value(ts=sample_ts, trait_df=trait_df)
        self.check_dimensions(genetic_result, sample_ts.num_individuals * num_trait)
        np.testing.assert_equal(
            genetic_result["individual_id"],
            np.tile(np.arange(sample_ts.num_individuals), 3),
        )
        np.testing.assert_equal(
            genetic_result["trait_id"],
            np.repeat(np.arange(3), sample_ts.num_individuals),
        )

    def test_additional_row(self, sample_ts, sample_df):
        """Check that adding unexpected rows to trait_df won't cause any errors"""
        df = sample_df.copy()
        df["height"] = [170, 180]
        tstrait.genetic_value(ts=sample_ts, trait_df=df)


class TestGenotype:
    """Test that genetic_value accurately detects individual genotypes."""

    def test_binary_tree(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 2, 3],
                "effect_size": [1, 10, 100],
                "trait_id": [0, 0, 0],
                "causal_allele": ["T", "C", "T"],
            }
        )

        ts = binary_tree()
        tree = ts.first()
        g0 = individual_genetic_values(tree, ts.site(0), "T", 1)
        g1 = individual_genetic_values(tree, ts.site(1), "T", 2)
        g2 = individual_genetic_values(tree, ts.site(2), "C", 3)
        g3 = individual_genetic_values(tree, ts.site(3), "C", 4)

        np.testing.assert_equal(g0, np.array([1, 0, 2]) * 1)
        np.testing.assert_equal(g1, np.array([1, 1, 0]) * 2)
        np.testing.assert_equal(g2, np.array([0, 1, 0]) * 3)
        np.testing.assert_equal(g3, np.array([1, 2, 0]) * 4)

        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 0],
                "individual_id": [0, 1, 2],
                "genetic_value": [101, 10, 202],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)

    def test_diff_ind_tree(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 2],
                "effect_size": [1, 10],
                "trait_id": [0, 0],
                "causal_allele": ["T", "C"],
            }
        )

        ts = diff_ind_tree()
        tree = ts.first()
        g0 = individual_genetic_values(tree, ts.site(0), "T", 1)
        g1 = individual_genetic_values(tree, ts.site(1), "T", 2)
        g2 = individual_genetic_values(tree, ts.site(2), "C", 3)
        g3 = individual_genetic_values(tree, ts.site(3), "C", 4)

        np.testing.assert_equal(g0, np.array([1, 1, 1]) * 1)
        np.testing.assert_equal(g1, np.array([1, 0, 1]) * 2)
        np.testing.assert_equal(g2, np.array([0, 1, 0]) * 3)
        np.testing.assert_equal(g3, np.array([1, 1, 1]) * 4)

        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)
        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 0],
                "individual_id": [0, 1, 2],
                "genetic_value": [1, 11, 1],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)

    def test_non_binary_tree(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 1],
                "effect_size": [1, 10],
                "trait_id": [0, 0],
                "causal_allele": ["A", "T"],
            }
        )

        ts = non_binary_tree()
        tree = ts.first()
        g0 = individual_genetic_values(tree, ts.site(0), "T", 1)
        g1 = individual_genetic_values(tree, ts.site(1), "C", 2)

        np.testing.assert_equal(g0, np.array([0, 1, 2]) * 1)
        np.testing.assert_equal(g1, np.array([0, 1, 1]) * 2)

        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 0],
                "individual_id": [0, 1, 2],
                "genetic_value": [2, 1, 10],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)

    def test_triploid(self):
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 1],
                "effect_size": [1, 10],
                "trait_id": [0, 0],
                "causal_allele": ["T", "C"],
            }
        )

        ts = triploid_tree()
        tree = ts.first()
        g0 = individual_genetic_values(tree, ts.site(0), "T", 1)
        g1 = individual_genetic_values(tree, ts.site(1), "C", 2)

        np.testing.assert_equal(g0, np.array([1, 2]) * 1)
        np.testing.assert_equal(g1, np.array([1, 1]) * 2)

        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)
        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0],
                "individual_id": [0, 1],
                "genetic_value": [11, 12],
            }
        )
        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)

    def test_allele_freq_one(self):
        ts = binary_tree()
        tables = ts.dump_tables()
        tables.sites.add_row(4, "A")
        tables.mutations.add_row(site=4, node=0, derived_state="T")
        tables.mutations.add_row(site=4, node=0, derived_state="A", parent=9)
        ts = tables.tree_sequence()
        trait_df = pd.DataFrame(
            {
                "site_id": [4],
                "effect_size": [1],
                "trait_id": [0],
                "causal_allele": ["A"],
            }
        )
        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)
        genetic_df = pd.DataFrame(
            {
                "trait_id": np.zeros(3),
                "individual_id": np.arange(3),
                "genetic_value": np.ones(3) * 2,
            }
        )
        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)


class TestTreeSeq:
    """Test the `genetic_value` function by using a tree sequence data."""

    def test_tree_seq(self):
        ts = binary_tree_seq()
        trait_df = pd.DataFrame(
            {
                "site_id": [0, 1, 2],
                "effect_size": [1, 10, 100],
                "causal_allele": ["T", "G", "T"],
                "trait_id": [0, 0, 0],
            }
        )

        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0],
                "individual_id": [0, 1],
                "genetic_value": [1, 22],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)

    def test_tree_seq_multiple_trait(self):
        ts = binary_tree_seq()
        trait_df = pd.DataFrame(
            {
                "trait_id": [0, 1, 0, 1],
                "site_id": [0, 0, 2, 2],
                "effect_size": [1, 2, 10, 20],
                "causal_allele": ["T", "T", "C", "C"],
            }
        )

        genetic_result = tstrait.genetic_value(ts=ts, trait_df=trait_df)

        genetic_df = pd.DataFrame(
            {
                "trait_id": [0, 0, 1, 1],
                "individual_id": [0, 1, 0, 1],
                "genetic_value": [11, 12, 22, 24],
            }
        )

        pd.testing.assert_frame_equal(genetic_df, genetic_result, check_dtype=False)


class TestNormaliseGenetic:
    def test_output(self, sample_ts):
        mean = 2
        var = 4
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=20, model=model, random_seed=500
        )
        genetic_df = tstrait.genetic_value(sample_ts, trait_df)

        normalised_df = tstrait.normalise_genetic_value(genetic_df, mean=mean, var=var)
        genetic_array = normalised_df["genetic_value"].values
        np.testing.assert_almost_equal(np.mean(genetic_array), mean, decimal=2)
        np.testing.assert_almost_equal(np.var(genetic_array, ddof=1), var, decimal=2)
        pd.testing.assert_series_equal(normalised_df["trait_id"], genetic_df["trait_id"])
        pd.testing.assert_series_equal(
            normalised_df["individual_id"], genetic_df["individual_id"]
        )

        num_ind = sample_ts.num_individuals
        assert len(normalised_df) == num_ind
        assert normalised_df.shape[1] == 3
        assert list(normalised_df.columns) == [
            "individual_id",
            "trait_id",
            "genetic_value",
        ]

    def test_default(self, sample_ts):
        mean = 0
        var = 1
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=20, model=model, random_seed=1000
        )
        genetic_df = tstrait.genetic_value(sample_ts, trait_df)
        normalised_df = tstrait.normalise_genetic_value(genetic_df)
        genetic_array = normalised_df["genetic_value"].values
        np.testing.assert_almost_equal(np.mean(genetic_array), mean, decimal=2)
        np.testing.assert_almost_equal(np.var(genetic_array, ddof=1), var, decimal=2)
        pd.testing.assert_series_equal(normalised_df["trait_id"], genetic_df["trait_id"])
        pd.testing.assert_series_equal(
            normalised_df["individual_id"], genetic_df["individual_id"]
        )

        num_ind = sample_ts.num_individuals
        assert len(normalised_df) == num_ind
        assert normalised_df.shape[1] == 3
        assert list(normalised_df.columns) == [
            "individual_id",
            "trait_id",
            "genetic_value",
        ]

    def test_column(self, sample_ts):
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=20, model=model, random_seed=1000
        )
        genetic_df = tstrait.genetic_value(sample_ts, trait_df)
        with pytest.raises(
            ValueError, match="columns must be included in genetic_df dataframe"
        ):
            tstrait.normalise_genetic_value(genetic_df[["trait_id", "individual_id"]])

        with pytest.raises(
            ValueError, match="columns must be included in genetic_df dataframe"
        ):
            tstrait.normalise_genetic_value(genetic_df[["trait_id", "genetic_value"]])

        with pytest.raises(
            ValueError, match="columns must be included in genetic_df dataframe"
        ):
            tstrait.normalise_genetic_value(
                genetic_df[["genetic_value", "individual_id"]]
            )

    @pytest.mark.parametrize("var", [0, -1])
    def test_negative_var(self, sample_ts, var):
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=20, model=model, random_seed=1000
        )
        genetic_df = tstrait.genetic_value(sample_ts, trait_df)

        with pytest.raises(ValueError, match="Variance must be greater than 0."):
            tstrait.normalise_genetic_value(genetic_df, var=var)

    @pytest.mark.parametrize("ddof", [0, 1])
    def test_ddof(self, sample_ts, ddof):
        model = tstrait.trait_model(distribution="normal", mean=2, var=6)
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=20, model=model, random_seed=1000
        )
        genetic_df = tstrait.genetic_value(sample_ts, trait_df)
        normalised_df = tstrait.normalise_genetic_value(
            genetic_df, mean=0, var=1, ddof=ddof
        )
        normalised_genetic_array = normalised_df["genetic_value"].values

        genetic_array = genetic_df["genetic_value"].values
        genetic_array = (genetic_array - np.mean(genetic_array)) / np.std(
            genetic_array, ddof=ddof
        )

        np.testing.assert_array_almost_equal(normalised_genetic_array, genetic_array)

    def test_pleiotropy(self, sample_ts):
        mean = 0
        var = 1
        model = tstrait.trait_model(
            distribution="multi_normal", mean=np.zeros(2), cov=np.identity(2)
        )
        trait_df = tstrait.sim_trait(
            ts=sample_ts, num_causal=20, model=model, random_seed=1000
        )
        genetic_df = tstrait.genetic_value(sample_ts, trait_df)
        normalised_df = tstrait.normalise_genetic_value(genetic_df, mean=mean, var=var)
        grouped = normalised_df.groupby(["trait_id"])[["genetic_value"]]
        mean_array = grouped.mean().values.T[0]
        var_array = grouped.var().values.T[0]
        np.testing.assert_almost_equal(mean_array, np.zeros(2), decimal=2)
        np.testing.assert_almost_equal(var_array, np.ones(2), decimal=2)


def naive_genetic_value(ts, trait_df, level):
    """Reference implementation of genetic_value, computed tree by tree.

    For each row of the trait dataframe this seeks to the tree at the causal
    site and propagates the causal allele down from the virtual root, stopping
    wherever a mutation changes the state, which is how genetic values were
    computed before the ARG descent.
    """
    num_trait = int(np.max(trait_df["trait_id"])) + 1
    size = {
        "individual": ts.num_individuals,
        "node": ts.num_nodes,
        "edge": ts.num_edges,
    }[level]
    genetic_value_table = np.zeros((num_trait, size))
    tree = tskit.Tree(ts)
    for data in trait_df.itertuples():
        site = ts.site(data.site_id)
        tree.seek(site.position)
        state = {tree.virtual_root: site.ancestral_state}
        has_mutation = set()
        for m in site.mutations:
            state[m.node] = m.derived_state
            has_mutation.add(m.node)
        # One extra entry, so that the virtual root can be a causal node.
        nodes_value = np.zeros(ts.num_nodes + 1)
        stack = [u for u, allele in state.items() if allele == data.causal_allele]
        while len(stack) > 0:
            u = stack.pop()
            nodes_value[u] = data.effect_size
            for child in tree.children(u):
                if child not in has_mutation:
                    stack.append(child)
        nodes_value = nodes_value[: ts.num_nodes]

        if level == "individual":
            value = np.zeros(ts.num_individuals)
            for u, individual in enumerate(ts.nodes_individual):
                if individual != tskit.NULL:
                    value[individual] += nodes_value[u]
        elif level == "edge":
            value = np.zeros(ts.num_edges)
            for u in range(ts.num_nodes):
                edge = tree.edge(u)
                if edge != tskit.NULL:
                    value[edge] += nodes_value[u]
        else:
            value = nodes_value
        genetic_value_table[data.trait_id] += value

    return pd.DataFrame(
        {
            "trait_id": np.repeat(np.arange(num_trait), size),
            f"{level}_id": np.tile(np.arange(size), num_trait),
            "genetic_value": genetic_value_table.flatten(),
        }
    )


class TestGeneticValueReference:
    """Compare genetic_value against a tree by tree reference implementation
    over a range of tree topologies and allele configurations.

    The causal alleles drawn by random_trait_df include the ancestral state of
    a site, which is the case the ARG descent folds into the value it seeds the
    roots with, so it is covered here rather than separately.
    """

    def verify(self, ts, num_trait, seed=1, levels=("node", "edge")):
        trait_df = random_trait_df(ts, num_trait, seed)
        for level in levels:
            if level == "individual" and ts.num_individuals == 0:
                continue
            expected = naive_genetic_value(ts, trait_df, level)
            result = tstrait.genetic_value(ts=ts, trait_df=trait_df, level=level)
            pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "ts_func",
        [
            binary_tree,
            diff_ind_tree,
            non_binary_tree,
            triploid_tree,
            binary_tree_seq,
            simple_tree_seq,
            allele_freq_one,
        ],
    )
    @pytest.mark.parametrize("num_trait", [1, 3])
    def test_data_tree_sequence(self, ts_func, num_trait):
        self.verify(ts_func(), num_trait, levels=("individual", "node", "edge"))

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    @pytest.mark.parametrize("rate", [2.0, 10.0])
    def test_all_trees(self, n, rate):
        ts = multi_allelic_mutations(all_trees_ts(n), rate=rate, seed=n)
        self.verify(ts, num_trait=2)

    @pytest.mark.parametrize("recombination_rate", [0, 1e-7])
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_simulated(self, recombination_rate, seed):
        ts = msprime.sim_ancestry(
            8,
            sequence_length=10_000,
            recombination_rate=recombination_rate,
            population_size=1000,
            random_seed=seed,
        )
        ts = multi_allelic_mutations(ts, rate=1e-4, seed=seed)
        self.verify(ts, num_trait=3, seed=seed, levels=("individual", "node", "edge"))

    def test_isolated_samples(self):
        """Isolated samples are roots, so they carry the ancestral state."""
        tables = tskit.TableCollection(sequence_length=10)
        for _ in range(4):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        site = tables.sites.add_row(position=0, ancestral_state="A")
        tables.mutations.add_row(site=site, node=0, derived_state="T")
        ts = tables.tree_sequence()
        for causal_allele in ["A", "T"]:
            trait_df = pd.DataFrame(
                {
                    "site_id": [0],
                    "effect_size": [2.5],
                    "trait_id": [0],
                    "causal_allele": [causal_allele],
                }
            )
            for level in ("node", "edge"):
                expected = naive_genetic_value(ts, trait_df, level)
                result = tstrait.genetic_value(ts=ts, trait_df=trait_df, level=level)
                pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_multiple_roots(self):
        """A forest has several roots, each carrying the ancestral state."""
        tables = tskit.Tree.generate_balanced(4).tree_sequence.dump_tables()
        tables.edges.replace_with(tables.edges[tables.edges.parent != 6])
        site = tables.sites.add_row(position=0, ancestral_state="A")
        tables.mutations.add_row(site=site, node=4, derived_state="T")
        tables.sort()
        ts = tables.tree_sequence()
        for causal_allele in ["A", "T"]:
            trait_df = pd.DataFrame(
                {
                    "site_id": [0],
                    "effect_size": [3.0],
                    "trait_id": [0],
                    "causal_allele": [causal_allele],
                }
            )
            for level in ("node", "edge"):
                expected = naive_genetic_value(ts, trait_df, level)
                result = tstrait.genetic_value(ts=ts, trait_df=trait_df, level=level)
                pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    def test_mutation_above_a_root(self):
        """A mutation with no edge applies to the root it sits above."""
        tables = tskit.Tree.generate_balanced(4).tree_sequence.dump_tables()
        site = tables.sites.add_row(position=0, ancestral_state="A")
        tables.mutations.add_row(site=site, node=6, derived_state="T")
        tables.mutations.add_row(site=site, node=4, derived_state="G")
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()
        for causal_allele in ["A", "T", "G"]:
            trait_df = pd.DataFrame(
                {
                    "site_id": [0],
                    "effect_size": [1.5],
                    "trait_id": [0],
                    "causal_allele": [causal_allele],
                }
            )
            for level in ("node", "edge"):
                expected = naive_genetic_value(ts, trait_df, level)
                result = tstrait.genetic_value(ts=ts, trait_df=trait_df, level=level)
                pd.testing.assert_frame_equal(result, expected, check_dtype=False)
