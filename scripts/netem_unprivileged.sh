#!/usr/bin/env bash
# Run a command as root of a private user, mount and network namespace, so the tc netem
# harness (live/netem.py, compare_modes.py --netem) works without sudo:
#
#   bash scripts/netem_unprivileged.sh "$PYTHON" scripts/compare_modes.py <clips> --netem ...
#
# Inside, the command is uid 0 with CAP_NET_ADMIN over its own network namespaces only:
# it can create the veth pair and the client namespace, and shape them with tc netem,
# but it cannot touch the host's network. The traffic still crosses the kernel's real
# TCP stack. /run is replaced by an empty tmpfs so `ip netns` has somewhere to keep
# its namespaces; files written elsewhere belong to the invoking user as usual.
#
# Needs unprivileged user namespaces (kernel.unprivileged_userns_clone, on by default
# on most distributions) and the sch_netem module.
set -euo pipefail
exec unshare --user --map-root-user --mount --net --propagation private \
    bash -c 'mount -t tmpfs none /run && mkdir -p /run/netns && ip link set lo up && exec "$@"' bash "$@"
