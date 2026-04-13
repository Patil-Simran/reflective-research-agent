from reflective_research.config.settings import Settings
from reflective_research.graph.nodes import make_route_after_verify


def test_route_after_verify_end_when_passed() -> None:
    route = make_route_after_verify(Settings(max_verification_revisions=2))
    assert route({"verification_passed": True, "revision_count": 0}) == "end"


def test_route_after_verify_revise_when_budget() -> None:
    route = make_route_after_verify(Settings(max_verification_revisions=2))
    assert route({"verification_passed": False, "revision_count": 0}) == "revise"


def test_route_after_verify_finalize_when_exhausted() -> None:
    route = make_route_after_verify(Settings(max_verification_revisions=2))
    assert route({"verification_passed": False, "revision_count": 2}) == "finalize"
