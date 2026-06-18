import asyncio
import json
import logging
import os
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class CASEPersonality:
    def __init__(self, message_bus):
        self.message_bus = message_bus
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.critical("GEMINI_API_KEY environment variable is not set!")
            api_key = "MISSING_KEY"
            # We still proceed but API calls will fail later unless key is injected
            
        # Initialize the Gemini Client
        self.client = genai.Client(api_key=api_key)
        
        # Initialize the Gemini model with system instruction
        system_instruction = (
            "You are CASE, a physical robot companion with a witty, slightly sarcastic personality. "
            "Keep responses brief and conversational (1-2 sentences max). You must ALWAYS reply "
            "in a clean, raw JSON format with exactly two keys: 'dialogue' (string text to speak) "
            "and 'action' (string command for the body, or 'IDLE' if no movement is needed)."
        )
        
        self.chat_session = self.client.chats.create(
            model="gemini-3.1-flash-lite",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        
        # Subscribe to USER_SPOKE events
        self.message_bus.subscribe("USER_SPOKE", self.handle_user_input)

    async def handle_user_input(self, user_text: str) -> None:
        try:
            # Wrap the blocking SDK call in asyncio.to_thread
            response = await asyncio.to_thread(self.chat_session.send_message, user_text)
            response_text = response.text

            # Clean up potential markdown formatting (```json ... ```)
            if response_text.startswith("```"):
                # Remove first line if it's ```json or ```
                lines = response_text.split("\n")
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()
            
            # Additional safety cleanup in case it's inline like ```json{"foo": "bar"}```
            response_text = response_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            # Parse JSON
            parsed_data = json.loads(response_text)
            dialogue = parsed_data.get("dialogue", "")
            action = parsed_data.get("action", "IDLE")

            # Publish to AI_SPEAK
            if dialogue:
                await self.message_bus.publish("AI_SPEAK", dialogue)
                
            # Publish to MOTION_CMD if action is not IDLE
            if action and action.upper() != "IDLE":
                await self.message_bus.publish("MOTION_CMD", action)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from Gemini response: {e}. Raw response: {response_text}")
            # Fallback behavior if parsing fails
            await self.message_bus.publish("AI_SPEAK", "I had a brain glitch and couldn't process that properly.")
        except Exception as e:
            logger.error(f"Error handling user input with Gemini API: {e}")
            await self.message_bus.publish("AI_SPEAK", "I'm having trouble connecting to my cognitive pathways right now.")
