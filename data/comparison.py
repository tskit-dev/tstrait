"""
This includes the Python codes to simulate quantitative traits based on three
external simulators, AlphaSimR, simplePHENOTYPES, and the simulation framework
described in the ARG-Needle paper.
"""
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    """
    Dataclass that contains simulated effect sizes and phenotypes.

    Attributes
    ----------
    trait : pandas.DataFrame
        Trait dataframe.
    phenotype : pandas.DataFrame
        Phenotype dataframe.
    """

    phenotype: pd.DataFrame
    trait: pd.DataFrame


def simplePHENOTYPES_simulation(
    ts, num_causal, add_effect, random_seed, num_trait=1, add_effect_2=1
):
    """
    The function to simulate quantitative traits by using simplePHENOTYPES.
    We will specify the number of causal sites and the parameter for the
    geometric series where the effect sizes are determined.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence data that will be used in the quantitative trait
        simulation.
    num_causal : int
        Number of causal sites.
    add_effect : float
        Parameter that is used in geometric series to obtain the effect
        sizes of the first trait.
    random_seed : int
        Random seed.
    num_trait : int
        Number of traits to be simulated. It must be 0 or 1.
    add_effect_2 : float
        Parameter that is used in geometric series to obtain the effect
        sizes of the second trait.
    """

    directory = tempfile.TemporaryDirectory()

    vcf_filename = "vcf_comparison_simplePHENOTYPES"
    with open(f"{directory.name}/{vcf_filename}.vcf", "w") as vcf_file:
        ts.write_vcf(vcf_file)
    cmd = ["Rscript", "data/simulate_simplePHENOTYPES.R"]
    args = [
        str(num_causal),
        str(num_trait),
        str(add_effect),
        str(add_effect_2),
        directory.name,
        vcf_filename,
        str(random_seed),
    ]
    input_cmd = cmd + args
    subprocess.check_output(input_cmd)

    if num_trait == 1:
        phenotype_df = pd.read_csv(
            f"{directory.name}/Simulated_Data_1_Reps_Herit_1.txt", sep="\t"
        )
        del phenotype_df["reps"]
        phenotype_df = phenotype_df.rename(
            columns={"<Trait>": "individual_id", "Pheno": "phenotype"}
        )
    else:
        phenotype_df = pd.read_csv(
            f"{directory.name}/Simulated_Data_1_Reps_Herit_1_1.txt", sep="\t"
        )
        del phenotype_df["Rep"]
        phenotype_df = pd.melt(
            phenotype_df,
            value_vars=["Trait_1_H2_1", "Trait_2_H2_1"],
            id_vars=["<Trait>"],
        )
        phenotype_df = phenotype_df.rename(
            columns={
                "<Trait>": "individual_id",
                "value": "phenotype",
                "variable": "trait_id",
            }
        )
        phenotype_df = phenotype_df.replace({"Trait_1_H2_1": 0, "Trait_2_H2_1": 1})

    num_ind = ts.num_individuals
    # Change the individual ID in simplePHENOTYPES output to be consistent with the
    # tstrait output
    for i in range(num_ind):
        phenotype_df = phenotype_df.replace(f"tsk_{i}", i)

    qtn_df = pd.read_csv(f"{directory.name}/Additive_Selected_QTNs.txt", sep="\t")

    # Obtain the list of causal allele
    causal_allele = []
    effect_size = []
    effect_size_2 = []
    for i, site_id in enumerate(qtn_df["snp"].values, start=1):
        # simplePHENOTYPES uses ancestral state as a causal allele
        allele = ts.site(site_id).ancestral_state
        causal_allele.append(allele)
        effect_size.append(add_effect**i)
        effect_size_2.append(add_effect_2**i)

    if num_trait == 2:
        effect_size = np.append(effect_size, effect_size_2)

    trait_df = pd.DataFrame(
        {
            "site_id": np.tile(qtn_df["snp"].values, num_trait),
            "causal_allele": np.tile(causal_allele, num_trait),
            "effect_size": effect_size,
            "trait_id": np.repeat(np.arange(num_trait), len(causal_allele)),
        }
    )

    simulation_result = SimulationResult(phenotype=phenotype_df, trait=trait_df)

    directory.cleanup()

    return simulation_result


def alphasimr_simulation(ts, num_causal, random_seed, corA=1, num_trait=1):
    """
    The function to simulate quantitative traits by using AlphaSimR. We will
    specify the number of causal sites, such that the AlphaSimR simulation
    will be conducted randomly.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence data that will be used in the quantitative trait
        simulation.
    num_causal : int
        Number of causal sites.
    corA : float
        Correlation of pleiotropic traits.
    num_trait : int
        Number of traits to be simulated. It must be 1 or 2.
    random_seed : int
        Random seed.

    Returns
    -------
    SimulationResult
        Dataclass object that includes simulated phenotypes and effect sizes.
    """

    directory = tempfile.TemporaryDirectory()
    tree_filename = "tree_comparison_AlphaSimR"
    ts.dump(f"{directory.name}/{tree_filename}.tree")
    phenotype_filename = "phenotype_comparison_AlphaSimR"
    trait_filename = "trait_comparison_AlphaSimR"
    cmd = ["Rscript", "data/simulate_AlphaSimR.R"]
    args = [
        str(num_causal),
        directory.name,
        tree_filename,
        phenotype_filename,
        trait_filename,
        str(corA),
        str(num_trait),
        str(random_seed),
    ]
    input_cmd = cmd + args
    subprocess.check_output(input_cmd)

    phenotype_df = pd.read_csv(f"{directory.name}/{phenotype_filename}.csv")
    trait_df = pd.read_csv(f"{directory.name}/{trait_filename}.csv")

    # Obtain the list of causal allele
    causal_allele = []
    for site_id in trait_df["site_id"]:
        allele = ts.mutation(site_id).derived_state
        causal_allele.append(allele)

    trait_df["causal_allele"] = causal_allele

    simulation_result = SimulationResult(phenotype=phenotype_df, trait=trait_df)

    directory.cleanup()

    return simulation_result


def arg_needle_simulation(ts, alpha, random_seed):
    """
    The function to simulate quantitative traits based on the simulation framework
    described in the ARG-Needle paper (Zhang et al., 2023). This assumes that all
    sites are causal. The codes are adapted from https://zenodo.org/records/7745746.

    Parameters
    ----------
    ts : tskit.TreeSequence
        The tree sequence data that will be used in the quantitative trait
        simulation.
    alpha : float
        The alpha parameter that will be used in frequency dependence
        architecture.
    random_seed : float
        Random seed.

    Returns
    -------
    SimulationResult
        Dataclass object that includes simulated phenotypes and effect sizes.
    """

    rng = np.random.default_rng(random_seed)
    num_ind = ts.num_individuals

    phenotypes = np.zeros(num_ind)
    beta_list = []
    causal_allele = []

    for variant in ts.variants():
        row = variant.genotypes.astype("float64")
        row = row.reshape((num_ind, 2)).sum(axis=-1)
        std = np.std(row, ddof=1)
        beta = rng.normal()
        beta_list.append(beta)
        causal_allele.append(variant.alleles[1])
        phenotypes += row * (beta * std**alpha)

    phenotypes -= np.mean(phenotypes)
    phenotypes /= np.std(phenotypes, ddof=1)

    phenotype_df = pd.DataFrame(
        {"individual_id": np.arange(len(phenotypes)), "phenotype": phenotypes}
    )
    trait_df = pd.DataFrame(
        {
            "site_id": np.arange(len(causal_allele)),
            "causal_allele": causal_allele,
            "effect_size": beta_list,
        }
    )

    simulation_result = SimulationResult(phenotype=phenotype_df, trait=trait_df)

    return simulation_result
