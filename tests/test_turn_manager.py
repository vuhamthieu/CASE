import unittest

from src.realtime.turn_manager import TurnManager, TurnState


class TurnManagerTests(unittest.TestCase):
    def test_state_transition_and_latency_breakdown(self):
        manager = TurnManager()
        manager.transition(TurnState.WAKE_ACK)
        self.assertEqual(manager.state, TurnState.WAKE_ACK)

        metrics = {
            "wake_detected_at": 1.0,
            "wake_ack_start_at": 1.1,
            "wake_ack_done_at": 1.5,
            "speech_started_at": 2.0,
            "last_speech_at": 3.2,
            "transcript_final_at": 3.45,
            "first_llm_chunk_at": 3.95,
            "first_tts_chunk_start_at": 4.0,
            "first_tts_chunk_done_at": 4.3,
            "first_audio_play_start_at": 4.35,
            "full_audio_done_at": 5.0,
        }
        breakdown = TurnManager.latency_breakdown(metrics)
        self.assertEqual(breakdown["wake_to_ack_start"], "0.100s")
        self.assertEqual(breakdown["ack_playback"], "0.400s")
        self.assertEqual(breakdown["last_speech_to_transcript_final"], "0.250s")
        self.assertEqual(breakdown["total_wake_to_first_audio"], "3.350s")


if __name__ == "__main__":
    unittest.main()
