"""What the server has sent that the network has not delivered yet.

The client learns of congestion only when delayed packets arrive, so its signal is
itself stuck in the queue. The sender knows at once how much it handed to the
network that has not been delivered: on a real socket that is the kernel's send
queue (SIOCOUTQ: unsent plus unacknowledged bytes); through the emulated link it is
the link's queue. Divided by the client's capacity estimate, it is the delay a new
frame would wait.

(Measuring the drain rate on the server from this backlog was tried and failed: the
backlog is only seen once per frame and often empties in between, so the rate came
out far too low and the guard then dropped nearly everything.)
"""
from __future__ import annotations

import fcntl
import socket
import struct
import termios

SIOCOUTQ = getattr(termios, 'TIOCOUTQ', 0x5411)  # the same request number on Linux


def socket_backlog(sock: socket.socket) -> int:
    try:
        return struct.unpack('i', fcntl.ioctl(sock.fileno(), SIOCOUTQ, b'\0' * 4))[0]
    except OSError:
        return 0


def backlog_bytes(sock: socket.socket, link=None) -> int:
    return (link.queued_bytes if link is not None else 0) + socket_backlog(sock)


# Offsets in Linux's struct tcp_info (include/uapi/linux/tcp.h), stable since 4.6.
_TCP_INFO = {'rto_us': (8, 'I'), 'mss': (16, 'I'), 'rtt_us': (68, 'I'), 'rttvar_us': (72, 'I'),
             'total_retrans': (100, 'I'), 'data_segs_out': (156, 'I')}


def tcp_path(sock: socket.socket) -> dict | None:
    """What the kernel's TCP knows about the path: smoothed RTT and its variation, the
    retransmission timeout, the segment size, and running counts of data segments sent
    and retransmitted (their ratio over a window is the loss rate). None where
    TCP_INFO is unavailable (not Linux, or not a TCP socket)."""
    try:
        raw = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 232)
    except (OSError, AttributeError):
        return None
    if len(raw) < 160:
        return None
    return {name: struct.unpack_from(kind, raw, offset)[0] for name, (offset, kind) in _TCP_INFO.items()}
