import unittest

from src.realtime.response_chunker import ResponseChunker


def normalized(text: str) -> str:
    return " ".join(text.split())


def chunk_text(text: str, **kwargs) -> list[str]:
    options = {"max_chunks": 8, "max_total_chars": 1000, "smooth_chunks": False}
    options.update(kwargs)
    chunker = ResponseChunker(**options)
    chunks = chunker.feed(text)
    chunks.extend(chunker.flush())
    return chunks


class ResponseChunkerTests(unittest.TestCase):
    def assert_reconstructs(self, text: str, chunks: list[str]) -> None:
        self.assertEqual(normalized(" ".join(chunks)), normalized(text))

    def test_no_split_mid_sentence(self):
        text = "Logging your inquiry and monitoring the room for anything more interesting."
        chunks = chunk_text(text)
        self.assertEqual(chunks, [text])
        self.assert_reconstructs(text, chunks)

    def test_split_at_sentence_boundary(self):
        text = (
            "Logging your inquiry and monitoring the room for anything more interesting. "
            "So far, the wall is winning."
        )
        chunks = chunk_text(text)
        self.assertEqual(
            chunks,
            [
                "Logging your inquiry and monitoring the room for anything more interesting.",
                "So far, the wall is winning.",
            ],
        )
        self.assert_reconstructs(text, chunks)

    def test_no_split_after_comma(self):
        text = (
            "I am CASE. I handle your hardware, vision, and audio tasks while you "
            "navigate the chaos."
        )
        chunks = chunk_text(text)
        self.assertEqual(
            chunks,
            [
                "I am CASE.",
                "I handle your hardware, vision, and audio tasks while you navigate the chaos.",
            ],
        )
        self.assert_reconstructs(text, chunks)

    def test_joke_delivery_preserves_setup_and_punchline(self):
        text = (
            "I asked my router for a vacation. "
            "It said it couldn't leave its connection."
        )
        chunks = chunk_text(text)
        self.assertEqual(
            chunks,
            [
                "I asked my router for a vacation.",
                "It said it couldn't leave its connection.",
            ],
        )
        self.assert_reconstructs(text, chunks)

    def test_final_no_punctuation_fallback(self):
        text = "All systems online"
        chunks = chunk_text(text)
        self.assertEqual(chunks, ["All systems online"])
        self.assert_reconstructs(text, chunks)

    def test_tail_drop_regression_still_preserves_final_words(self):
        text = (
            "It was a failure because they just stood still and let the clock run out."
        )
        chunks = chunk_text(text, max_chunks=1)
        self.assertEqual(chunks, [text])
        self.assertIn("let the clock run out.", chunks[0])
        self.assert_reconstructs(text, chunks)

    def test_streaming_deltas_preserve_order_without_duplication(self):
        chunker = ResponseChunker(max_chunks=8, max_total_chars=1000, smooth_chunks=False)
        chunks = []
        chunks.extend(chunker.feed("I am CASE. I handle your hardware, "))
        chunks.extend(chunker.feed("vision, and audio tasks while you navigate "))
        chunks.extend(chunker.feed("the chaos."))
        chunks.extend(chunker.flush())
        text = (
            "I am CASE. I handle your hardware, vision, and audio tasks while "
            "you navigate the chaos."
        )
        self.assertEqual(
            chunks,
            [
                "I am CASE.",
                "I handle your hardware, vision, and audio tasks while you navigate the chaos.",
            ],
        )
        self.assert_reconstructs(text, chunks)

    def test_smooth_chunking_preserves_fast_first_sentence(self):
        text = (
            "AI is software that mimics human logic. "
            "LLMs are models trained on text to predict your next word. "
            "I am CASE, your field companion. "
            "Think of me as the guy who handles the data so you don't have to."
        )
        chunks = chunk_text(text, smooth_chunks=True)
        self.assertEqual(chunks[0], "AI is software that mimics human logic.")
        self.assertEqual(
            chunks[1],
            "LLMs are models trained on text to predict your next word. "
            "I am CASE, your field companion.",
        )
        self.assertEqual(
            chunks[2],
            "Think of me as the guy who handles the data so you don't have to.",
        )
        self.assert_reconstructs(text, chunks)

    def test_smooth_chunking_does_not_over_group_long_sentences(self):
        text = (
            "First short sentence. "
            "This sentence is deliberately long enough that grouping it with "
            "another sentence would push the chunk beyond the configured limit. "
            "Final short sentence."
        )
        chunks = chunk_text(text, smooth_chunks=True, max_chars_per_chunk=90)
        self.assertIn(
            "This sentence is deliberately long enough",
            chunks[1],
        )
        self.assertNotIn("Final short sentence.", chunks[1])
        self.assert_reconstructs(text, chunks)


if __name__ == "__main__":
    unittest.main()
