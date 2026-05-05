import unittest

import twopy


class ImportTest(unittest.TestCase):
    def test_version_is_available(self) -> None:
        self.assertEqual(twopy.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
