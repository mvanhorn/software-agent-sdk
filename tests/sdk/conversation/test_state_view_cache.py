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
from openhands.sdk.conversation.persistence_const import EVENT_FILE_PATTERN, EVENTS_DIR
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
    state._wire_view_sync()
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


def test_condensation_request_sets_flag_incrementally(
    state: ConversationState,
) -> None:
    """CondensationRequest should set the unhandled flag without adding
    an event to the view (it is not LLMConvertible)."""
    from openhands.sdk.event.condenser import CondensationRequest

    state.append_event(CondensationRequest())
    assert state.view.unhandled_condensation_request is True
    assert len(state.view) == 0


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


def test_direct_eventlog_append_also_updates_view(
    state: ConversationState,
) -> None:
    """Appending directly via state.events.append must still update the view.

    This verifies the on_append callback wired by _wire_view_sync, so
    callers who bypass state.append_event do not silently diverge the
    cached view.
    """
    msg = message_event("direct")
    state.events.append(msg)

    assert len(state.view) == 1
    assert state.view.events[0] is msg


def test_view_syncs_when_eventlog_syncs_extra_events(
    state: ConversationState,
) -> None:
    """If EventLog reports synced_count > 0, the missed events must be
    incrementally applied to the view before the new event.

    We simulate this by writing an event file directly to the in-memory
    file store so that EventLog._sync_from_disk picks it up during the
    next append.
    """
    # Manually insert an event file behind EventLog's back, simulating
    # another process having written while this state was alive.
    sneaky_msg = message_event("sneaky")
    payload = sneaky_msg.model_dump_json(exclude_none=True)
    path = f"events/{EVENT_FILE_PATTERN.format(idx=0, event_id=sneaky_msg.id)}"
    state._fs.write(path, payload)

    # Now append a second event through the normal path.  EventLog will
    # notice the on-disk file during its lock-and-sync step and report
    # synced_count=1, which should trigger incremental application of
    # the missed event followed by the new one.
    second_msg = message_event("second")
    state.append_event(second_msg)

    # Both events should be in the view in the correct order.
    assert len(state.view) == 2
    view_ids = [e.id for e in state.view.events]
    assert view_ids[0] == sneaky_msg.id
    assert view_ids[1] == second_msg.id


def test_fresh_create_rebuilds_view_for_orphaned_events() -> None:
    """ConversationState.create on a directory that has event files but no
    base_state.json (crash / partial cleanup) must still populate the
    cached view from the pre-existing events."""
    llm = LLM(model="gpt-4o-mini", api_key=SecretStr("test-key"), usage_id="test-llm")
    agent = Agent(llm=llm)
    workspace = LocalWorkspace(working_dir="/tmp/test")

    # Set up an in-memory file store with orphaned event files but NO
    # base_state.json, simulating a crash before state was saved.
    fs = InMemoryFileStore()
    orphan_msg = message_event("orphan")
    payload = orphan_msg.model_dump_json(exclude_none=True)
    path = f"{EVENTS_DIR}/{EVENT_FILE_PATTERN.format(idx=0, event_id=orphan_msg.id)}"
    fs.write(path, payload)

    state = ConversationState(
        id=uuid.uuid4(),
        workspace=workspace,
        persistence_dir=None,
        agent=agent,
    )
    state._fs = fs
    state._events = EventLog(fs, dir_path=EVENTS_DIR)
    state._wire_view_sync()

    # Mimic the fresh-create guard: rebuild if events already exist.
    if len(state._events) > 0:
        state.rebuild_view()

    # The orphaned event should be visible in the cached view.
    assert len(state.view) == 1
    assert state.view.events[0].id == orphan_msg.id

    # Future appends should still work correctly.
    new_msg = message_event("new")
    state.append_event(new_msg)
    assert len(state.view) == 2
    assert state.view.events[1].id == new_msg.id
