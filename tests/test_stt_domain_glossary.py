import unittest
from pathlib import Path

from src.stt_backends.domain_glossary import DomainGlossary


class SttDomainGlossaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.glossary = DomainGlossary.from_file(
            Path(__file__).resolve().parents[1] / "config" / "stt_domain_glossary.json"
        )

    def assertRepair(self, transcript, expected):
        repaired, match = self.glossary.repair(transcript)
        self.assertEqual(repaired, expected)
        self.assertIsNotNone(match)

    def test_repairs_game_and_hardware_terms(self):
        cases = {
            "do you know g p a six": "do you know GTA 6",
            "what is gpa 6": "what is GTA 6",
            "tell me about grand theft auto six": "tell me about GTA 6",
            "do you know point cheap alto": "do you know Grand Theft Auto",
            "e s p thirty two": "ESP32",
            "p c a nine six eight five": "PCA9685",
        }
        for transcript, expected in cases.items():
            with self.subTest(transcript=transcript):
                self.assertRepair(transcript, expected)

    def test_gpa_academic_context_is_unchanged(self):
        for transcript in ("what is my GPA", "how to calculate GPA", "grade point average"):
            with self.subTest(transcript=transcript):
                repaired, match = self.glossary.repair(transcript)
                self.assertEqual(repaired, transcript)
                self.assertIsNone(match)

    def test_uk_case_robot_context_repairs_to_case_name(self):
        for transcript in ("UK case", "thank you UK case", "are you listening UK case"):
            with self.subTest(transcript=transcript):
                repaired, match = self.glossary.repair(transcript)
                self.assertIn("you, CASE", repaired)
                self.assertIsNotNone(match)

    def test_uk_legal_or_court_case_is_unchanged(self):
        for transcript in ("UK legal case", "UK court case", "tell me about the UK court case"):
            with self.subTest(transcript=transcript):
                repaired, match = self.glossary.repair(transcript)
                self.assertEqual(repaired, transcript)
                self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
