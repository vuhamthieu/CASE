import unittest

from src.realtime.response_chunker import ResponseChunker


class ResponseChunkerTests(unittest.TestCase):
    def test_chunks_at_sentence_boundaries(self):
        chunker = ResponseChunker(
            min_chars=25,
            max_chars=90,
            max_chunks=4,
            max_total_chars=360,
        )
        chunks = []
        for delta in (
            "I'm CASE. I handle speech, vision, and hardware control. ",
            "Basically, a field robot with questionable job security.",
        ):
            chunks.extend(chunker.feed(delta))
        chunks.extend(chunker.flush())
        self.assertEqual(
            chunks,
            [
                "I'm CASE. I handle speech, vision, and hardware control. Basically, a field robot with questionable job security.",
            ],
        )

    def test_first_chunk_emits_before_full_response_arrives(self):
        chunker = ResponseChunker(
            min_chars=25,
            max_chars=90,
            max_chunks=4,
            max_total_chars=360,
            merge_tiny_chunks=False,
        )
        early = chunker.feed("I'm CASE. ")
        self.assertEqual(early, ["I'm CASE."])
        later = chunker.feed("I handle voice, vision, and hardware control. ")
        self.assertEqual(later, ["I handle voice, vision, and hardware control."])

    def test_length_chunks_do_not_end_on_weak_words(self):
        chunker = ResponseChunker(
            min_chars=35,
            max_chars=55,
            absolute_max_chars=70,
            max_chunks=4,
            max_total_chars=220,
        )
        chunks = chunker.feed(
            "This response should split near a safe word boundary with the "
            "remaining text continuing after the break."
        )
        chunks.extend(chunker.flush())
        for chunk in chunks[:-1]:
            self.assertNotRegex(chunk.lower(), r"\b(the|a|an|to|of|and|but|or|with|in|on|your|my|is|are)$")

    def test_holds_dependent_continuation_until_sentence_end(self):
        chunker = ResponseChunker(
            min_chars=35,
            max_chars=95,
            absolute_max_chars=130,
            max_chunks=4,
            max_total_chars=420,
        )
        chunks = chunker.feed(
            "I processed his request so quickly that I figured out exactly how much "
            "he was underpaid before he finished"
        )
        self.assertEqual(chunks, [])
        chunks.extend(chunker.feed(" his sentence."))
        chunks.extend(chunker.flush())
        self.assertEqual(
            chunks,
            [
                "I processed his request so quickly that I figured out exactly how much "
                "he was underpaid before he finished his sentence."
            ],
        )

    def test_longer_joke_uses_natural_sentence_chunks(self):
        chunker = ResponseChunker(
            min_chars=35,
            max_chars=95,
            absolute_max_chars=130,
            max_chunks=4,
            max_total_chars=420,
        )
        chunks = chunker.feed(
            "A technician once tried to overclock my logic core to make me faster. "
            "I processed his request so quickly that I calculated his salary before "
            "he finished the sentence. He did not ask again."
        )
        chunks.extend(chunker.flush())
        self.assertEqual(
            chunks,
            [
                "A technician once tried to overclock my logic core to make me faster.",
                "I processed his request so quickly that I calculated his salary before he finished the sentence. He did not ask again.",
            ],
        )
        self.assertTrue(all(chunk.endswith((".", "?", "!")) for chunk in chunks))

    def test_short_response_under_single_chunk_limit_merges(self):
        chunker = ResponseChunker(
            min_chars=35,
            max_chars=110,
            absolute_max_chars=160,
            max_chunks=4,
            max_total_chars=420,
            single_chunk_under_chars=130,
        )
        chunks = chunker.feed(
            "I am not a boy. I am hardware with standards. Try again."
        )
        chunks.extend(chunker.flush())
        self.assertEqual(
            chunks,
            ["I am not a boy. I am hardware with standards. Try again."],
        )

    def test_tiny_final_chunk_merges_with_previous_when_short(self):
        chunker = ResponseChunker(
            min_chars=35,
            max_chars=110,
            absolute_max_chars=160,
            max_chunks=4,
            max_total_chars=420,
            single_chunk_under_chars=130,
        )
        chunks = chunker.feed(
            "This is a short correction with one last tiny sentence. Try again."
        )
        chunks.extend(chunker.flush())
        self.assertEqual(
            chunks,
            ["This is a short correction with one last tiny sentence. Try again."],
        )

    def test_joke_setup_punchline_can_remain_two_chunks(self):
        chunker = ResponseChunker(
            min_chars=35,
            max_chars=70,
            absolute_max_chars=90,
            max_chunks=4,
            max_total_chars=420,
            single_chunk_under_chars=130,
        )
        chunks = chunker.feed(
            "Why did the robot go on a diet? It had too many bytes."
        )
        chunks.extend(chunker.flush())
        self.assertIn(
            chunks,
            [
                ["Why did the robot go on a diet? It had too many bytes."],
                ["Why did the robot go on a diet?", "It had too many bytes."],
            ],
        )

    def test_max_chunks_and_total_chars_are_enforced(self):
        chunker = ResponseChunker(
            min_chars=10,
            max_chars=40,
            max_chunks=2,
            max_total_chars=80,
        )
        chunks = chunker.feed(
            "First sentence is allowed. Second sentence is allowed. "
            "Third sentence should not be emitted."
        )
        chunks.extend(chunker.flush())
        self.assertLessEqual(len(chunks), 2)
        self.assertLessEqual(sum(len(chunk) for chunk in chunks), 80)


if __name__ == "__main__":
    unittest.main()
