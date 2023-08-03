import tstrait
from tstrait.provenance import __version__


class TestVersion:
    def test_version(self):
        assert tstrait.__version__ == __version__
