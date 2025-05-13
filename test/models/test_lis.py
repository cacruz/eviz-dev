from os import path, environ
from eviz.lib.autoviz_base import Autoviz
import pytest
import unittest


@pytest.mark.skip_integration
class TestBasic(unittest.TestCase):
    def setUp(self):
        model = 'lis'
        self.eviz = Autoviz([model])
        cfg = path.join(environ['EVIZ_CONFIG_PATH'], model, model+'.yaml')
        self.eviz._set_config(model, config_file=cfg, config_dir=None)

    def test_run(self):
        self.eviz.run_model()


if __name__ == '__main__':
    unittest.main()
