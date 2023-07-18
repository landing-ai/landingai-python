import time

import pytest

from landingai.timer import Timer


def test_timer(caplog):
    t = Timer(name="manual")
    t.start()
    time.sleep(0.1)
    t.stop()
    assert "Timer 'manual' finished. Elapsed time:" in caplog.text

    with Timer(name="context manager"):
        time.sleep(0.2)
    assert "Timer 'context manager' finished. Elapsed time:" in caplog.text

    @Timer(name="decorator")
    def do_stuff():
        time.sleep(0.3)

    do_stuff()
    assert "Timer 'decorator' finished. Elapsed time:" in caplog.text

    with Timer():
        time.sleep(0.2)
    assert "Timer 'default' finished. Elapsed time:" in caplog.text


def test_timer_get_global_stats():
    timer_keys = ["1", "2", "3"]
    for k in timer_keys:
        for _ in range(2 * int(k)):
            with Timer(name=k):
                time.sleep(0.01 * int(k))

    for k in timer_keys:
        actual = Timer.stats.stats(k)
        assert actual["count"] == int(k) * 2
        assert actual["min"] == pytest.approx(0.01 * int(k), abs=0.01)
        assert actual["max"] == pytest.approx(0.01 * int(k), abs=0.01)
        assert actual["sum_total"] >= 0.01 * int(k)
