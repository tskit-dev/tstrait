import numbers

import numpy as np

from .base import _check_dataframe


class _EnvSimulator:
    """Simulator class to simulate environmental noise of individuals.

    Parameters
    ----------
    genetic_df : pandas.DataFrame
        Pandas dataframe that includes genetic value of individuals. It must include
        individual ID, genetic value and trait ID.
    h2 : float or array-like
        Narrow-sense heritability.
    random_seed : int
        The random seed.
    """

    def __init__(self, genetic_df, h2, random_seed):
        self.genetic_df = genetic_df[["trait_id", "individual_id", "genetic_value"]]
        self.h2 = h2
        self.rng = np.random.default_rng(random_seed)

    def _sim_env(self, var, h2):
        """Simulate environmental noise based on variance and narrow-sense
        heritability
        """
        env_std = np.sqrt((1 - h2) / h2 * var)
        env_noise = self.rng.normal(loc=0.0, scale=env_std)

        return env_noise

    def _sim_environment(self):
        """Simulate environmental values based on genetic values of individuals and
        narrow-sense heritability
        """
        df = self.genetic_df.copy()
        h2_array = np.take(self.h2, self.genetic_df.trait_id)

        grouped = df.groupby("trait_id", sort=False)["genetic_value"]
        var_array = grouped.transform("var")

        df["environmental_noise"] = self._sim_env(var_array, h2_array)

        df["phenotype"] = df["genetic_value"] + df["environmental_noise"]

        return df


def sim_env(genetic_df, *, h2=None, random_seed=None):
    """
    Simulates environmental noise.

    Parameters
    ----------
    genetic_df : pandas.DataFrame
        Genetic value dataframe.
    h2 : float or array-like, default None.
        Narrow-sense heritability. When it is 1, environmental noise will be a vector of
        zeros. If `h2` is array-like, the dimension of `h2` must match the number of
        traits to be simulated. If None, h2 will be 1.
    random_seed : int, default None
        Random seed of simulation. If None, simulation will be conducted randomly.

    Returns
    -------
    pandas.DataFrame
        Dataframe with simulated environmental noise.

    Raises
    ------
    ValueError
        If `h2` <= 0 or `h2` > 1

    See Also
    --------
    sim_genetic : Return a dataclass with genetic value dataframe, which can be used as
        `genetic_df` input.

    Notes
    -----
    The `genetic_df` input has some requirements that will be noted below.

    1. Columns

    The following columns must be included in `genetic_df`:

        * **trait_id**: Trait ID.
        * **individual_id**: Individual ID inside the tree sequence input.
        * **genetic_value**: Simulated genetic values.

    2. Data requirement

    Trait IDs in **trait_id** column must start from 0 and be consecutive.

    The dataframe output has the following columns:

        * **trait_id**: Trait ID.
        * **individual_id**: Individual ID inside the tree sequence input.
        * **genetic_value**: Simulated genetic values.
        * **environmental_noise**: Simulated environmental noise.
        * **phenotype**: Simulated phenotype.

    Examples
    --------
    See :ref:`environment_noise` for worked examples.
    """
    genetic_df = _check_dataframe(
        genetic_df, ["trait_id", "individual_id", "genetic_value"], "genetic_df"
    )

    trait_id = genetic_df["trait_id"].unique()

    if np.min(trait_id) != 0 or np.max(trait_id) != len(trait_id) - 1:
        raise ValueError("trait_id must be consecutive and start from 0")

    h2 = 1 if h2 is None else h2
    if isinstance(h2, numbers.Real):
        h2 = np.ones(len(trait_id)) * h2

    if len(h2) != len(trait_id):
        raise ValueError("Length of h2 must match the number of traits")

    if np.min(h2) <= 0 or np.max(h2) > 1:
        raise ValueError("Narrow-sense heritability must be 0 < h2 <= 1")

    simulator = _EnvSimulator(
        genetic_df=genetic_df,
        h2=h2,
        random_seed=random_seed,
    )

    phenotype_df = simulator._sim_environment()

    return phenotype_df
