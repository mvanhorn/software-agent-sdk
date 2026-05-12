"""Workflow controls (Plan / Verify / Save) for the agent.

These live in their own leaf module — separate from ``settings.model`` — so
that ``openhands.sdk.conversation.request`` can carry an ``AgentControls`` on
``StartConversationRequest`` without creating a cycle through
``settings.model`` (which already imports ``SendMessageRequest`` from
``conversation.request``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from openhands.sdk.settings.metadata import (
    SETTINGS_METADATA_KEY,
    SettingProminence,
    SettingsFieldMetadata,
)


PlanLevel = Literal["none", "some", "lots"]
"""How much research the agent should do before starting work."""

VerifyLevel = Literal["none", "some", "lots"]
"""How much testing / QA the agent should do when finishing."""

SaveMode = Literal["worktree", "local", "push", "pr", "pr_ready", "merge"]
"""How the agent should deliver its work when done."""


class AgentControls(BaseModel):
    """High-level controls that govern how the agent approaches a task.

    The three controls — :attr:`plan`, :attr:`verify`, and :attr:`save` — are
    workflow knobs the user can dial in for a conversation. Defaults are
    persisted in agent-server settings; the user can change them mid-
    conversation, after which the new values are shipped to the agent on every
    subsequent user message (see :meth:`render_active_block`).
    """

    plan: PlanLevel = Field(
        default="some",
        description=(
            "How much research/analysis to do before starting work. "
            "``none`` = jump in immediately, ``some`` = look around a bit, "
            "``lots`` = do extensive research."
        ),
        json_schema_extra={
            SETTINGS_METADATA_KEY: SettingsFieldMetadata(
                label="Plan",
                prominence=SettingProminence.MAJOR,
            ).model_dump()
        },
    )
    verify: VerifyLevel = Field(
        default="some",
        description=(
            "How much testing/QA to do when finishing. "
            "``none`` = skip tests/lint, ``some`` = basic lint and affected "
            "tests, ``lots`` = full test suite."
        ),
        json_schema_extra={
            SETTINGS_METADATA_KEY: SettingsFieldMetadata(
                label="Verify",
                prominence=SettingProminence.MAJOR,
            ).model_dump()
        },
    )
    save: SaveMode = Field(
        default="worktree",
        description=(
            "How to deliver the work when done. "
            "``worktree`` = keep on the local worktree, ``local`` = move "
            "branch into main workspace, ``push`` = push to remote, "
            "``pr`` = open a pull request, ``pr_ready`` = open a PR and "
            "iterate until CI / review pass, ``merge`` = open a PR and "
            "merge once mergeable."
        ),
        json_schema_extra={
            SETTINGS_METADATA_KEY: SettingsFieldMetadata(
                label="Save",
                prominence=SettingProminence.MAJOR,
            ).model_dump()
        },
    )

    def render_active_block(self) -> str:
        """Render the per-turn ``<ACTIVE_CONTROLS>`` block.

        Shipped alongside every user message so the agent always reads the
        current control values next to the current task — see
        ``LocalConversation.send_message`` for the injection point and
        ``system_prompt.j2`` (CONTROLS section) for the catalog the agent
        interprets these against.
        """
        return (
            "<ACTIVE_CONTROLS>\n"
            f"plan={self.plan}; verify={self.verify}; save={self.save}\n"
            "</ACTIVE_CONTROLS>"
        )
