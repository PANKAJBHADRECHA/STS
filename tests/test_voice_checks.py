import unittest
from scripts.voice_checks import check


class SpeechScoringTests(unittest.TestCase):
    def test_wrong_result_is_not_accepted_because_operand_matches(self):
        self.assertFalse(check("spoken_interruption", "6 + 6 = 12."))

    def test_spoken_number_answer_is_understood(self):
        self.assertTrue(check("spoken_interruption", "Three plus three equals six."))

    def test_week_count_does_not_select_one_week(self):
        self.assertTrue(check("slow_speech", "There are seven days in one week."))

    def test_memory_denial_fails(self):
        self.assertFalse(check("memory", "I don't remember that.", "remember"))

    def test_negated_place_is_not_a_recall_pass(self):
        self.assertFalse(
            check("recall", "Alex does not live in London.", r"(?=.*alex)(?=.*london)")
        )

    def test_bold_number_is_not_a_false_failure(self):
        self.assertTrue(check("slow_speech", "There are **7** days in one week."))
