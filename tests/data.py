import tskit


def binary_tree():
    """Sample tree sequence. The genotype is the following:
    Genotype: [Node0, Node1, Node2, Node3, Node4, Node5]
    Node4 and Node5 are internal nodes
    Ancestral State: A
    Site0: [A, A, T, T; A, T], has 1 mutation
    Site1: [C, T, A, A; T, A], has 2 mutations of the same type
    Site2: [C, A, A, A; A, A], has multiple mutations in the same node
    Site3: [C, C, T, T; C, T], has reverse mutation and mutation in head node

    Individual1: Node0 and Node1
    Individual2: Node2 and Node3
    Individual0: Node4 and Node5 (Internal node)
    """
    ts = tskit.Tree.generate_balanced(4, span=10).tree_sequence
    tables = ts.dump_tables()
    for j in range(4):
        tables.sites.add_row(j, "A")

    tables.individuals.add_row()
    tables.individuals.add_row()
    tables.individuals.add_row()
    individuals = tables.nodes.individual
    individuals[[0, 1]] = 1
    individuals[[2, 3]] = 2
    individuals[[4, 5]] = 0
    tables.nodes.individual = individuals

    tables.mutations.add_row(site=0, node=5, derived_state="T")

    tables.mutations.add_row(site=1, node=4, derived_state="T")
    tables.mutations.add_row(site=1, node=0, derived_state="C", parent=1)

    tables.mutations.add_row(site=2, node=0, derived_state="T")
    tables.mutations.add_row(site=2, node=0, derived_state="G", parent=3)
    tables.mutations.add_row(site=2, node=0, derived_state="T", parent=4)
    tables.mutations.add_row(site=2, node=0, derived_state="C", parent=5)

    tables.mutations.add_row(site=3, node=6, derived_state="T")
    tables.mutations.add_row(site=3, node=4, derived_state="C", parent=7)

    ts = tables.tree_sequence()

    return ts


def diff_ind_tree():
    """Same mutation and tree structure as binary_tree, but individuals will
    have different nodes. See the details of the tree sequence in the docstring
    for `binary_tree`.

    Individual1: Node0 and Node2
    Individual2: Node1 and Node3
    Individual0: Node4 and Node5 (Internal Node)
    """
    ts = binary_tree()
    tables = ts.dump_tables()
    tables.individuals.clear()
    tables.individuals.add_row()
    tables.individuals.add_row()
    tables.individuals.add_row()
    individuals = tables.nodes.individual
    individuals[[0, 2]] = 1
    individuals[[1, 3]] = 2
    individuals[[4, 5]] = 0
    tables.nodes.individual = individuals
    ts = tables.tree_sequence()
    return ts


def non_binary_tree():
    """Non-binary tree sequence.
    Genotype: [Node0, Node1, Node2, Node3, Node4, Node5]
    Ancestral State: A
    Site0: [A, A, A, T, T, T], has 1 mutation
    Site1: [A, A, A, C, C, T], has reverse mutation and multiple mutations
        in the same node

    Individual0: Node0 and Node1
    Individual1: Node2 and Node3
    Individual2: Node4 and Node5
    """
    ts = tskit.Tree.generate_balanced(6, arity=4, span=10).tree_sequence

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

    tables.mutations.add_row(site=0, node=6, derived_state="T")
    tables.mutations.add_row(site=1, node=6, derived_state="T")
    tables.mutations.add_row(site=1, node=6, derived_state="C", parent=1)
    tables.mutations.add_row(site=1, node=5, derived_state="T", parent=2)

    ts = tables.tree_sequence()
    return ts


def triploid_tree():
    """Same mutation and tree structure as non_binary_tree, but individuals have
    different nodes and are triploids. See the details of the tree sequence
    in the doctring for `non_binary_tree`.

    Individual0: Node0, Node2 and Node4
    Individual1: Node1, Node3 and Node5
    """
    ts = non_binary_tree()
    tables = ts.dump_tables()
    tables.individuals.clear()
    tables.individuals.add_row()
    tables.individuals.add_row()
    individuals = tables.nodes.individual
    individuals[[0, 2, 4]] = 0
    individuals[[1, 3, 5]] = 1
    tables.nodes.individual = individuals
    ts = tables.tree_sequence()
    return ts


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


def binary_tree_seq():
    """Sample tree sequence
    Ancestral State: A
    Genotype: [Node0, Node1, Node2, Node3]
    Site0: [A, T, T, T], has 1 mutation
    Site1: [A, T, G, G], has reverse mutation
    Site2: [A, C, C, A], has multiple mutation on the same edge

    Individual0: Node0 and Node1
    Individual1: Node2 and Node3
    """
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

    tables.sites.add_row(7, "A")
    tables.sites.add_row(11, "A")
    tables.sites.add_row(12, "A")

    tables.mutations.add_row(site=0, node=4, derived_state="T")
    tables.mutations.add_row(site=1, node=5, derived_state="T")
    tables.mutations.add_row(site=1, node=4, derived_state="G", parent=1)
    tables.mutations.add_row(site=2, node=4, derived_state="T")
    tables.mutations.add_row(site=2, node=4, derived_state="C", parent=3)

    ts = tables.tree_sequence()

    return ts


def simple_tree_seq():
    """
    Tree sequence data with a single mutation in each tree.
    Ancestral State: A
    Genotype: [Node0, Node1, Node2, Node3]
    Site0: [A, T, T, T]
    Site1: [A, A, A, A]
    Site2: [C, A, A, A]

    Individual0: Node0 and Node1
    Individual1: Node2 and Node3
    """
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

    tables.sites.add_row(7, "A")
    tables.sites.add_row(11, "A")
    tables.sites.add_row(12, "A")

    tables.mutations.add_row(site=0, node=4, derived_state="T")
    tables.mutations.add_row(site=1, node=4, derived_state="A")
    tables.mutations.add_row(site=2, node=0, derived_state="C")

    ts = tables.tree_sequence()

    return ts


def allele_freq_one():
    """
    Sample tree sequence with allele frequence one at a single site
    with ancestral state A.
    """
    ts = tskit.Tree.generate_balanced(4, span=10).tree_sequence
    tables = ts.dump_tables()
    tables.sites.add_row(0, "A")

    tables.individuals.add_row()
    tables.individuals.add_row()
    tables.individuals.add_row()
    individuals = tables.nodes.individual
    individuals[[0, 1]] = 1
    individuals[[2, 3]] = 2
    individuals[[4, 5]] = 0
    tables.nodes.individual = individuals

    tables.mutations.add_row(site=0, node=5, derived_state="T")
    tables.mutations.add_row(site=0, node=5, derived_state="A", parent=0)

    ts = tables.tree_sequence()

    return ts
