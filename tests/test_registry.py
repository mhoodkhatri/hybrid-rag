"""SessionRegistry expiry and LRU eviction, driven by a fake clock (no Streamlit runtime)."""

from __future__ import annotations

from app import SessionEntry, SessionRegistry


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeIndex:
    def __init__(self) -> None:
        self.deleted = False

    def delete(self) -> None:
        self.deleted = True

    def __len__(self) -> int:
        return 1


def _entry() -> SessionEntry:
    return SessionEntry(FakeIndex(), files={"a.pdf": b"%PDF"})  # type: ignore[arg-type]


def test_idle_entry_expires_and_frees_memory() -> None:
    clock = FakeClock()
    reg = SessionRegistry(ttl_s=60, max_sessions=10, clock=clock)
    e = _entry()
    reg.put("s1", e)
    clock.now += 59
    assert reg.sweep() == [] and reg.get("s1") is e
    clock.now += 2  # 61 s idle
    assert reg.sweep() == ["s1"]
    assert reg.get("s1") is None
    assert e.index.deleted and e.files == {}


def test_touch_keeps_entry_alive() -> None:
    clock = FakeClock()
    reg = SessionRegistry(ttl_s=60, max_sessions=10, clock=clock)
    reg.put("s1", _entry())
    for _ in range(5):
        clock.now += 50
        reg.touch("s1")
        assert reg.sweep() == []
    assert reg.get("s1") is not None


def test_touch_unknown_session_is_noop() -> None:
    reg = SessionRegistry(ttl_s=60, max_sessions=10, clock=FakeClock())
    reg.touch("nobody")
    assert len(reg) == 0


def test_cap_evicts_least_recently_used() -> None:
    clock = FakeClock()
    reg = SessionRegistry(ttl_s=3600, max_sessions=2, clock=clock)
    a, b, c = _entry(), _entry(), _entry()
    reg.put("a", a)
    clock.now += 1
    reg.put("b", b)
    clock.now += 1
    reg.touch("a")  # a is now more recent than b
    clock.now += 1
    reg.put("c", c)  # over the cap: b is the least recently used
    assert reg.get("b") is None and b.index.deleted
    assert reg.get("a") is a and reg.get("c") is c
    assert not a.index.deleted and not c.index.deleted


def test_remove_deletes_index() -> None:
    reg = SessionRegistry(ttl_s=60, max_sessions=10, clock=FakeClock())
    e = _entry()
    reg.put("s1", e)
    reg.remove("s1")
    reg.remove("s1")  # second remove is harmless
    assert reg.get("s1") is None and e.index.deleted


def test_delete_failure_does_not_break_sweep() -> None:
    clock = FakeClock()
    reg = SessionRegistry(ttl_s=10, max_sessions=10, clock=clock)

    class Broken(FakeIndex):
        def delete(self) -> None:
            raise RuntimeError("chroma hiccup")

    reg.put("bad", SessionEntry(Broken()))  # type: ignore[arg-type]
    reg.put("good", _entry())
    clock.now += 11
    assert sorted(reg.sweep()) == ["bad", "good"]
    assert len(reg) == 0
