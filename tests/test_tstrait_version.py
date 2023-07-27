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
