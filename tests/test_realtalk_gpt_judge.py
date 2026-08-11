import unittest

from src.experiments.realtalk_gpt_judge import _parse_bool, _parse_empathy


class RealTalkGptJudgeTest(unittest.TestCase):
    def test_boolean_parser(self):
        self.assertTrue(_parse_bool("True"))
        self.assertFalse(_parse_bool("'False'."))
        self.assertTrue(_parse_bool("Grounding: True"))

    def test_empathy_parser(self):
        self.assertEqual(
            _parse_empathy('{"emotional_reaction":1,"interpretation":2,"exploration":0}'),
            {"emotional_reaction": 1, "interpretation": 2, "exploration": 0},
        )

    def test_empathy_parser_rejects_range(self):
        with self.assertRaises(ValueError):
            _parse_empathy('{"emotional_reaction":3,"interpretation":0,"exploration":0}')


if __name__ == "__main__":
    unittest.main()
