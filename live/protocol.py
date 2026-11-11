"""Framing between the live server and client, over one TCP connection.

Server to client: H (JSON header), P (one encoded frame), R (JSON bitrate of every
rung over the last second, sent at each keyframe), G (JSON: the backlog guard dropped
frames up to the next keyframe), E (JSON end of stream).
Client to server: S (JSON switch request, for example {"rung": "x2@28"}), C (JSON
capacity estimate in kbit/s, for the server's backlog guard).

A rung is a rendition at a CRF, written 'x2@28'. A bare rendition ('x2') means that
rendition at the stream's default CRF.

Each record is a 1-byte kind and a 4-byte big-endian length, then the payload.
Only for trusted peers: payloads are not authenticated.
"""
from __future__ import annotations

import json
import socket
import struct
import threading
from typing import NamedTuple

RENDITIONS = ('direct', 'x2', 'x4')
SCALE = {'direct': 1, 'x2': 2, 'x4': 4}
LABEL = {'direct': 'gt', 'x2': 'lr2', 'x4': 'lr4'}  # filter-graph outputs
MAX_RECORD = 8 * 1024 * 1024
_RECORD = struct.Struct('!cI')
_PACKET = struct.Struct('!BIBBd')  # rendition, frame index, keyframe, crf, server send time


def rung_name(rendition: str, crf: int) -> str:
    return f'{rendition}@{crf}'


def parse_rung(text: str, default_crf: int | None = None) -> tuple[str, int]:
    """'x2@28' -> ('x2', 28); 'x2' -> ('x2', default_crf)."""
    rendition, _, crf = text.partition('@')
    if rendition not in SCALE:
        raise ValueError(f'unknown rendition {rendition!r}')
    if not crf:
        if default_crf is None:
            raise ValueError(f'{text!r} needs a CRF')
        return rendition, default_crf
    return rendition, int(crf)


class Packet(NamedTuple):
    rendition: str
    index: int
    keyframe: bool
    crf: int
    sent: float  # server monotonic time; the same clock as the client only on one machine
    data: bytes


class Channel:
    """Thread-safe record writer and a reader over a connected socket."""

    def __init__(self, sock: socket.socket, link=None) -> None:
        """link: an EmulatedLink that carries what this end sends, instead of the socket directly."""
        self.sock = sock
        self.link = link
        self.reader = sock.makefile('rb')
        self.lock = threading.Lock()
        self.bytes_sent = 0
        self.bytes_received = 0

    def send(self, kind: bytes, payload: bytes) -> int:
        record = _RECORD.pack(kind, len(payload)) + payload
        with self.lock:
            if self.link is not None:
                self.link.send(record)
            else:
                self.sock.sendall(record)
            self.bytes_sent += len(record)
        return len(record)

    def send_json(self, kind: bytes, value: dict) -> int:
        return self.send(kind, json.dumps(value).encode())

    def send_packet(self, packet: Packet) -> int:
        head = _PACKET.pack(RENDITIONS.index(packet.rendition), packet.index, packet.keyframe,
                            packet.crf, packet.sent)
        return self.send(b'P', head + packet.data)

    def receive(self) -> tuple[bytes, bytes] | None:
        """Next (kind, payload), or None when the peer closed the connection."""
        head = self._exact(_RECORD.size)
        if head is None:
            return None
        kind, size = _RECORD.unpack(head)
        if size > MAX_RECORD:
            raise ValueError(f'record of {size} bytes is too large')
        payload = self._exact(size)
        if payload is None:
            raise EOFError('connection closed inside a record')
        self.bytes_received += _RECORD.size + size
        return kind, payload

    def _exact(self, size: int) -> bytes | None:
        data = self.reader.read(size)
        if not data and size:
            return None
        if len(data) != size:
            raise EOFError(f'expected {size} bytes, got {len(data)}')
        return data


def parse_packet(payload: bytes) -> Packet:
    rendition, index, keyframe, crf, sent = _PACKET.unpack_from(payload)
    return Packet(RENDITIONS[rendition], index, bool(keyframe), crf, sent, payload[_PACKET.size:])
