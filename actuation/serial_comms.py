import asyncio
import serial
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from middleware.message_bus import AsyncMessageBus

logger = logging.getLogger(__name__)

class SerialBridge:
    def __init__(self, bus: 'AsyncMessageBus', port: str = '/dev/serial0', baudrate: int = 115200):
        self.bus = bus
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self._buffer: str = ""

        try:
            self.serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=0)
            logger.info(f"Successfully connected to serial port {self.port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            logger.warning(f"Failed to open serial port {self.port}: {e}. Running in degraded mode without hardware link.")

        # Subscribe to outgoing commands from the brain
        self.bus.subscribe("MOTION_CMD", self.transmit_command)

    async def transmit_command(self, cmd_text: str) -> None:
        if self.serial and self.serial.is_open:
            try:
                payload = f"{cmd_text}\n".encode('utf-8')
                self.serial.write(payload)
                logger.info(f"[UART TX] -> {cmd_text}")
            except Exception as e:
                logger.error(f"Failed to write to serial port: {e}")
        else:
            logger.warning(f"Discarding [UART TX] -> {cmd_text} (Serial port not connected)")

    async def listen_loop(self) -> None:
        while True:
            if self.serial and self.serial.is_open:
                try:
                    if self.serial.in_waiting > 0:
                        raw_data = self.serial.read(self.serial.in_waiting)
                        decoded = raw_data.decode('utf-8', errors='ignore')
                        self._buffer += decoded

                        while '\n' in self._buffer:
                            line, self._buffer = self._buffer.split('\n', 1)
                            line = line.strip()
                            if line:
                                logger.info(f"[UART RX] <- {line}")
                                await self.bus.publish("TELEMETRY", line)
                except Exception as e:
                    logger.error(f"Error reading from serial port: {e}")

            # CRITICAL: Yield control back to the async event loop
            await asyncio.sleep(0.01)
