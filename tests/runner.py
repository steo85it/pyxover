# tests/runner.py
import unittest

# import your test modules
# import bela_pipel
# import mla_pipel
# import lidar_pipel

# initialize the test suite
from tests import mla_pipel, lidar_pipel, bela_pipel

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(mla_pipel))
suite.addTests(loader.loadTestsFromModule(lidar_pipel))
suite.addTests(loader.loadTestsFromModule(bela_pipel))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)