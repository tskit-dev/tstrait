import pkgutil

import tstrait
from packaging.version import Version


class TestPythonVersion:
    """
    Test that the version is PEP440 compliant
    """

    def test_version(self):
        assert (
            str(Version(tstrait._version.tstrait_version))
            == tstrait._version.tstrait_version
        )
        assert tstrait.__version__ == tstrait._version.tstrait_version


PUBLIC_MODULE = [
    "tstrait.base",
    "tstrait.simulate_effect_size",
    "tstrait.simulate_phenotype",
    "tstrait.trait_model",
]


def unexpected_module(name):
    if "._" in name or ".tests" in name or ".setup" in name:
        return False

    elif name in PUBLIC_MODULE:
        return False

    else:
        return True


class Test_module:
    def test_unexpected_module(self):
        modnames = []
        for _, modname, _ispkg in pkgutil.walk_packages(
            path=tstrait.__path__, prefix=tstrait.__name__ + ".", onerror=None
        ):
            if unexpected_module(modname):
                modnames.append(modname)

        if modnames:
            raise AssertionError(f"Found unexpected modules: {modnames}")
