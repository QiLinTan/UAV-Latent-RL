"""Single-slot non-blocking buffer for high-to-low reference packets."""

from __future__ import annotations

from .reference_packet import ReferencePacket


class AsyncBuffer:
    """Keep only the newest accepted packet; never make the reader wait."""

    def __init__(self):
        self._packet: ReferencePacket | None = None
        self.accepted_packets = 0
        self.rejected_packets = 0
        self.rejected_versions = 0

    @property
    def latest_packet(self) -> ReferencePacket | None:
        return self._packet

    def clear(self) -> None:
        self._packet = None

    def commit(self, packet: ReferencePacket) -> bool:
        """Atomically accept a strictly newer packet.

        A late result from an older upper-control computation cannot overwrite a
        reference that was already committed by a newer result.
        """

        if not isinstance(packet, ReferencePacket):
            raise TypeError("AsyncBuffer accepts ReferencePacket values only.")
        if self._packet is not None and int(packet.version) <= int(self._packet.version):
            self.rejected_packets += 1
            self.rejected_versions += 1
            return False
        self._packet = packet
        self.accepted_packets += 1
        return True

    def latest_valid(self, current_time: float) -> ReferencePacket | None:
        packet = self._packet
        if packet is None or not packet.is_valid(current_time):
            return None
        return packet

    def packet_age(self, current_time: float) -> float | None:
        if self._packet is None:
            return None
        return self._packet.packet_age(current_time)
