import unittest
from pathlib import Path

from src.experiments.realtalk_gpt_judge import (
    REFLECTIVENESS_PROMPT,
    GROUNDING_PROMPT,
    _contexts,
    _parse_bool,
    _parse_empathy,
)


class RealTalkGptJudgeTest(unittest.TestCase):
    def test_prompts_include_appendix_c_examples(self):
        self.assertIn("I did what I thought was best", REFLECTIVENESS_PROMPT)
        self.assertIn("Can you tell me more", GROUNDING_PROMPT)

    def test_context_is_limited_to_target_session(self):
        rows = [{"speaker": "Akib", "result_id": "akib:message_16:session_2:turn_1"}]
        context = _contexts(Path("dataset"), rows)[rows[0]["result_id"]]
        self.assertNotIn("Good morning. How's it going?", context)

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
