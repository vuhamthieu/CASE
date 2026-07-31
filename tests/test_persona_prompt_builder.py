import unittest

from src.memory.session_memory import SessionMemory
from src.persona.case_prompt_builder import CasePromptBuilder


class PersonaPromptBuilderTests(unittest.TestCase):
    def test_voice_prompt_requests_concise_spoken_replies(self):
        memory = SessionMemory()
        result = CasePromptBuilder().build(
            user_text="Tell me a joke.",
            memory_context=memory.context(),
        )

        self.assertIn("style=concise_spoken", result.prompt)
        self.assertIn("use 1-2 short spoken sentences", result.prompt)
        self.assertIn("jokes use at most 2 short sentences", result.prompt)
        self.assertIn("avoid paragraph-length answers", result.prompt)


if __name__ == "__main__":
    unittest.main()
