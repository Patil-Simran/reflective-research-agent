from reflective_research.config.settings import Settings
from reflective_research.graph.nodes import make_route_after_reflect


def test_route_stops_when_max_gather_reached() -> None:
    route = make_route_after_reflect(Settings(evidence_brief_enabled=True))
    state = {
        "need_more": True,
        "gather_count": 3,
        "max_iterations": 3,
    }
    assert route(state) == "brief"


def test_route_stops_at_synthesize_when_brief_disabled() -> None:
    route = make_route_after_reflect(Settings(evidence_brief_enabled=False))
    state = {
        "need_more": True,
        "gather_count": 3,
        "max_iterations": 3,
    }
    assert route(state) == "synthesize"


def test_route_loops_when_under_cap() -> None:
    route = make_route_after_reflect(Settings())
    state = {
        "need_more": True,
        "gather_count": 1,
        "max_iterations": 3,
    }
    assert route(state) == "gather"


def test_route_brief_when_not_need_more_and_brief_on() -> None:
    route = make_route_after_reflect(Settings(evidence_brief_enabled=True))
    state = {
        "need_more": False,
        "gather_count": 1,
        "max_iterations": 3,
    }
    assert route(state) == "brief"


def test_route_synthesize_when_not_need_more_and_brief_off() -> None:
    route = make_route_after_reflect(Settings(evidence_brief_enabled=False))
    state = {
        "need_more": False,
        "gather_count": 1,
        "max_iterations": 3,
    }
    assert route(state) == "synthesize"
