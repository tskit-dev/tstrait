"""
Tests for edge effects and edge, node, and individual genetic values.

The tree sequence is based on the recombining ARG in
docs/individual_node_edge.md. The expected values can be calculated by hand.
"""

import io

import numpy as np
import pandas as pd
import pytest
import tskit

import tstrait


@pytest.fixture(scope="module")
def individual_node_edge_ts():
    """
    Return the small recombining tree sequence from the documentation.

    Individuals 0 and 1 are diploid:

        individual 0: nodes 0 and 1
        individual 1: nodes 2 and 3

    Node 1 has two immediately ancestral edges because of recombination:

        edge 1: [0, 5), parent 4 -> child 1
        edge 2: [5, 10), parent 5 -> child 1
    """
    individuals = io.StringIO(
        """\
flags location parents
    0        0      -1
    0        0      -1
"""
    )

    nodes = io.StringIO(
        """\
id is_sample time individual
 0         1    0          0
 1         1    0          0
 2         1    0          1
 3         1    0          1
 4         0    1         -1
 5         0    1         -1
 6         0    2         -1
 7         0    4         -1
"""
    )

    edges = io.StringIO(
        """\
left right parent child
   0    10      4     0
   0     5      4     1
   5    10      5     1
   0    10      5     2
   0    10      6     3
   0    10      6     5
   0    10      7     4
   0    10      7     6
"""
    )

    sites = io.StringIO(
        """\
position ancestral_state
       0               0
       1               0
       2               0
       3               0
       4               0
       5               0
       6               0
       7               0
       8               0
       9               0
"""
    )

    mutations = io.StringIO(
        """\
site node time derived_state parent
   0    4  3.0             1     -1
   1    3  1.5             1     -1
   2    1  0.5             1     -1
   4    4  2.0             1     -1
   6    6  3.0             1     -1
   7    5  1.5             1     -1
   8    3  1.0             1     -1
   9    1  0.5             1     -1
"""
    )

    ts = tskit.load_text(
        individuals=individuals,
        nodes=nodes,
        edges=edges,
        sites=sites,
        mutations=mutations,
        strict=False,
    )

    # Check if tree sequence is as we expect it to be
    assert ts.sequence_length == 10
    assert ts.num_trees == 2
    assert ts.num_individuals == 2
    assert ts.num_nodes == 8
    assert ts.num_edges == 8
    assert ts.num_sites == 10
    assert ts.num_mutations == 8

    return ts


def make_trait_df(effect_sizes):
    """Return a causal-allele effect table for the fixed set of sites."""
    site_ids = [0, 1, 2, 4, 6, 7, 8, 9]

    return pd.DataFrame(
        {
            "site_id": site_ids,
            "effect_size": effect_sizes,
            "trait_id": [0] * len(site_ids),
            "causal_allele": ["1"] * len(site_ids),
        }
    )


def assert_value_frame(
    observed,
    id_column,
    value_column,
    expected_values,
):
    """Check the IDs, columns, and values of a single-trait output."""
    assert list(observed.columns) == [
        "trait_id",
        id_column,
        value_column,
    ]

    observed = observed.sort_values(["trait_id", id_column]).reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "trait_id": np.zeros(len(expected_values), dtype=int),
            id_column: np.arange(len(expected_values)),
            value_column: np.asarray(expected_values, dtype=float),
        }
    )

    pd.testing.assert_frame_equal(
        observed,
        expected,
        check_dtype=False,
    )


def test_edge_node_and_individual_genetic_values(individual_node_edge_ts):
    """
    Check the complete edge-effect to individual-value calculation.

    The sequence of calculations is:

        causal-allele effects
            -> edge effects
            -> edge genetic values
            -> node genetic values
            -> individual genetic values
    """
    ts = individual_node_edge_ts
    # See values in docs/individual_node_edge.md.
    effect_sizes = [1, 1, -1, 1, -1, -1, -2, 1]
    expected_edge_effects = [0, -1, 1, 0, -1, -1, 2, -1]
    expected_edge_values = [2, 1, -1, -2, -2, -2, 2, -1]
    expected_node_values = [2, 0, -2, -2, 2, -2, -1, 0]
    expected_individual_values = [2, -4]

    trait_df = make_trait_df(effect_sizes)

    edge_effect_df = tstrait.edge_effect(ts, trait_df)
    edge_value_df = tstrait.genetic_value(
        ts,
        trait_df,
        level="edge",
    )
    node_value_df = tstrait.genetic_value(
        ts,
        trait_df,
        level="node",
    )
    individual_value_df = tstrait.genetic_value(
        ts,
        trait_df,
        level="individual",
    )

    assert_value_frame(
        edge_effect_df,
        id_column="edge_id",
        value_column="effect_size",
        expected_values=expected_edge_effects,
    )
    assert_value_frame(
        edge_value_df,
        id_column="edge_id",
        value_column="genetic_value",
        expected_values=expected_edge_values,
    )
    assert_value_frame(
        node_value_df,
        id_column="node_id",
        value_column="genetic_value",
        expected_values=expected_node_values,
    )
    assert_value_frame(
        individual_value_df,
        id_column="individual_id",
        value_column="genetic_value",
        expected_values=expected_individual_values,
    )

    # The default level must return individual genetic values.
    default_value_df = tstrait.genetic_value(ts, trait_df)

    pd.testing.assert_frame_equal(
        individual_value_df,
        default_value_df,
        check_dtype=False,
    )

    # Independently calculate sample-node values from the genotype matrix.
    # This does not use tstrait's edge or node accumulation algorithm.
    effects_by_site = np.zeros(ts.num_sites)
    for row in trait_df.itertuples(index=False):
        effects_by_site[int(row.site_id)] += float(row.effect_size)

    sample_values = ts.genotype_matrix().T @ effects_by_site

    np.testing.assert_allclose(
        sample_values,
        np.asarray(expected_node_values[:4], dtype=float),
    )

    # Individual 0 consists of sample nodes 0 and 1.
    # Individual 1 consists of sample nodes 2 and 3.
    genotype_based_individual_values = np.array(
        [
            sample_values[0] + sample_values[1],
            sample_values[2] + sample_values[3],
        ]
    )

    np.testing.assert_allclose(
        genotype_based_individual_values,
        np.asarray(expected_individual_values, dtype=float),
    )
