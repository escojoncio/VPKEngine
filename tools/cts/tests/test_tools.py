#!/usr/bin/env python3
"""Checks of tools/cts that need no deqp: python3 -I tools/cts/tests/test_tools.py"""
import json, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CTS = os.path.dirname(HERE)
sys.path.insert(0, CTS)
import run_lot  # noqa: E402


class TailText(unittest.TestCase):
    def test_last_bytes_of_a_big_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "big.log")
            with open(p, "w") as fh:
                for i in range(200000):
                    fh.write(f"line {i}\n")
            t = run_lot.tail_text(p, 1500)
            self.assertLessEqual(len(t), 1500)
            self.assertTrue(t.endswith("line 199999\n"))

    def test_small_file_whole(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.log")
            open(p, "w").write("a\nb\n")
            self.assertEqual(run_lot.tail_text(p, 1500), "a\nb\n")


class Diff(unittest.TestCase):
    def run_diff(self, old, new):
        with tempfile.TemporaryDirectory() as d:
            for name, cases in (("old.json", old), ("new.json", new)):
                json.dump({"cases": cases}, open(os.path.join(d, name), "w"))
            out = os.path.join(d, "diff.md")
            subprocess.run([sys.executable, "-I", os.path.join(CTS, "diff.py"), os.path.join(d, "old.json"),
                            os.path.join(d, "new.json"), "--out", out], check=True, capture_output=True)
            return open(out).read()

    def test_not_run_before_and_failing_now_is_reported(self):
        # The previous shard ran out of budget before reaching "b"; now it runs and fails: that must not vanish.
        md = self.run_diff({"a": "Pass", "b": "NotRun"}, {"a": "Pass", "b": "Fail"})
        self.assertIn("new cases failing: 1", md)

    def test_regression_still_counted(self):
        md = self.run_diff({"a": "Pass"}, {"a": "Fail"})
        self.assertIn("regressions: 1", md)


if __name__ == "__main__":
    unittest.main()
