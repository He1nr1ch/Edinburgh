"""The suite runs the same on any machine: no test reaches the network or the user's files."""

import socket

import pytest


@pytest.mark.unit
@pytest.mark.parametrize("connect", [
    lambda: socket.create_connection(("192.0.2.1", 80), timeout=1),
    lambda: socket.socket().connect_ex(("192.0.2.1", 80)),
    lambda: socket.getaddrinfo("example.com", 443),
], ids=["connect", "connect_ex", "dns"])
def test_a_test_cannot_reach_the_network(connect):
    """A test that silently depends on a live vendor passes or fails with the
    machine it runs on; conftest refuses the connection instead."""
    with pytest.raises(OSError, match="reach the network"):
        connect()


@pytest.mark.unit
@pytest.mark.parametrize("key", ["results_dir", "data_cache_dir", "memory_log_path"])
def test_the_suite_keeps_its_files_out_of_the_users_home(key):
    """A test that reads the user's cache sees what their runs fetched, and passes
    or fails with it; one that writes there changes their data."""
    from pathlib import Path

    from tradingagents.default_config import DEFAULT_CONFIG

    assert not Path(DEFAULT_CONFIG[key]).is_relative_to(Path.home() / ".tradingagents")
