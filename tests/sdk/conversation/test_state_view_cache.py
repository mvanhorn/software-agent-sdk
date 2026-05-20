"""Tests for the cached `ConversationState.view` and the `append_event` /
`rebuild_view` write paths added for issue #3053.

These tests assert that the incremental view stays in sync with the event log
without paying the cost of `View.from_events` on every append, and that the
full `enforce_properties` pass is reserved for explicit `rebuild_view` calls
(cold load, fork, error recovery).
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import SecretStr

from openhands.sdk import LLM, Agent
from openhands.sdk.context.view import View
from openhands.sdk.conversation.event_store import EventLog
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.io import InMemoryFileStore
from openhands.sdk.workspace import LocalWorkspace
from tests.sdk.context.view.conftest import message_event


@pytest.fixture
def state() -> ConversationState:
    """A bare ConversationState with an in-memory event log attached.

    We do not use `ConversationState.create` here because that path also
    touches a LocalFileStore on disk; for these unit tests an in-memory
    store is sufficient.
    """
    llm = LLM(model="gpt-4o-mini", api_key=SecretStr("test-key"), usage_id="test-llm")
    agent = Agent(llm=llm)
    workspace = LocalWorkspace(working_dir="/tmp/test")

    state = ConversationState(
        id=uuid.uuid4(),
        workspace=workspace,
        persistence_dir=None,
        agent=agent,
    )
    state._fs = InMemoryFileStore()
    state._events = EventLog(state._fs)
    return state


def test_fresh_state_has_empty_view(state: ConversationState) -> None:
    assert len(state.view) == 0
    assert state.view.events == []


def test_append_event_updates_both_event_log_and_view(
    state: ConversationState,
) -> None:
    msg = message_event("hello")
    state.append_event(msg)

    # EventLog rehydrates from disk on read, so we compare by id rather than
    # identity for the underlying log; the view holds direct references and
    # can be compared with `is`.
    assert len(state.events) == 1
    assert state.events[0].id == msg.id
    assert len(state.view) == 1
    assert state.view.events[0] is msg


def test_view_stays_in_parity_with_from_events_after_many_appends(
    state: ConversationState,
) -> None:
    msgs = [message_event(f"msg {i}") for i in range(5)]
    for msg in msgs:
        state.append_event(msg)

    rebuilt = View.from_events(state.events)
    assert [e.id for e in state.view.events] == [e.id for e in rebuilt.events]
    assert (
        state.view.unhandled_condensation_request
        == rebuilt.unhandled_condensation_request
    )


def test_condensation_event_is_applied_incrementally(
    state: ConversationState,
) -> None:
    msgs = [message_event(f"msg {i}") for i in range(3)]
    for msg in msgs:
        state.append_event(msg)

    condensation = Condensation(
        forgotten_event_ids={msgs[0].id, msgs[2].id},
        llm_response_id="resp_1",
    )
    state.append_event(condensation)

    # The Condensation is still in the underlying log (it is not LLM-convertible
    # but is part of the persisted history); the view, however, should reflect
    # the condensation by dropping the forgotten messages.
    assert len(state.view) == 1
    assert state.view.events[0] is msgs[1]


def test_append_event_does_not_run_enforce_on_hot_path(
    state: ConversationState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hot path must not pay the cost of `enforce_properties`.

    This is the core perf invariant from #3053: incremental updates rely on
    manipulation indices keeping the view well-formed, so enforcement should
    only run on `rebuild_view`.
    """
    call_count = 0
    original_enforce = View.enforce_properties

    def counting_enforce(self: View, all_events):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        return original_enforce(self, all_events)

    monkeypatch.setattr(View, "enforce_properties", counting_enforce)

    for i in range(10):
        state.append_event(message_event(f"msg {i}"))

    assert call_count == 0, (
        "append_event must not invoke enforce_properties on the hot path"
    )


def test_rebuild_view_runs_enforce(
    state: ConversationState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rebuild_view is the one place where full property enforcement runs."""
    call_count = 0
    original_enforce = View.enforce_properties

    def counting_enforce(self: View, all_events):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        return original_enforce(self, all_events)

    monkeypatch.setattr(View, "enforce_properties", counting_enforce)

    for i in range(3):
        state.append_event(message_event(f"msg {i}"))

    # Sanity: no enforce so far.
    assert call_count == 0

    state.rebuild_view()
    assert call_count >= 1
    # Parity check after the rebuild.
    assert [e.id for e in state.view.events] == [e.id for e in state.events]


def test_rebuild_view_replaces_cached_instance(state: ConversationState) -> None:
    """rebuild_view should produce a fresh View instance derived from the log."""
    state.append_event(message_event("hello"))
    before = state.view
    state.rebuild_view()
    after = state.view

    assert before is not after
    assert [e.id for e in after.events] == [e.id for e in before.events]
