from app.services.rate_limit import InMemoryRateLimiter


def test_allows_up_to_limit_then_blocks():
    t = [1000.0]
    rl = InMemoryRateLimiter(limit=3, window_seconds=60, clock=lambda: t[0])
    assert [rl.allow("k") for _ in range(3)] == [True, True, True]
    assert rl.allow("k") is False


def test_window_resets():
    t = [0.0]
    rl = InMemoryRateLimiter(limit=2, window_seconds=10, clock=lambda: t[0])
    assert rl.allow("k") and rl.allow("k")
    assert rl.allow("k") is False
    t[0] = 11.0  # advance past the window
    assert rl.allow("k") is True


def test_keys_are_independent():
    t = [0.0]
    rl = InMemoryRateLimiter(limit=1, window_seconds=60, clock=lambda: t[0])
    assert rl.allow("a") is True
    assert rl.allow("b") is True
    assert rl.allow("a") is False
