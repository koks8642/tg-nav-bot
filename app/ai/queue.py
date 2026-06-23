"""A small fair queue for outgoing persona replies.

Two lanes — DIRECT (someone addressed the persona) and AMBIENT (an entity was
mentioned / random butt-in). DIRECT is favoured, but AMBIENT is never starved:
after a short streak of DIRECT picks an AMBIENT one is served if waiting. Stale
jobs (the conversation has moved on) are dropped instead of answered late.

Pure logic (no awaits, no time source of its own) so it is fully unit-tested;
the worker injects ``now``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .decision import AMBIENT, DIRECT


@dataclass
class Job:
    chat_id: int
    reply_to: int          # message id the bot replies to (the trigger msg)
    user_id: int | None
    username: str | None
    text: str
    priority: str          # DIRECT | AMBIENT
    enqueued_at: float
    plan: dict[str, Any]    # current ReplyPlan, JSON-safe for tracing
    replied_to: int | None = None  # msg id the USER replied to (if any)
    needs_classifier: bool = False
    persona_key: str = ""
    profile_version: str = ""
    thread_id: int = 0


class FairQueue:
    def __init__(self, *, lane_max: int = 30, stale_sec: float = 45.0,
                 direct_streak_max: int = 2):
        self._direct: deque[Job] = deque()
        self._ambient: deque[Job] = deque()
        self.lane_max = lane_max
        self.stale_sec = stale_sec
        self.direct_streak_max = direct_streak_max
        self._streak = 0

    def __len__(self) -> int:
        return len(self._direct) + len(self._ambient)

    def push(self, job: Job) -> None:
        lane = self._direct if job.priority == DIRECT else self._ambient
        lane.append(job)
        while len(lane) > self.lane_max:
            lane.popleft()  # drop the oldest in this lane under flood

    def clear(self) -> int:
        removed = len(self)
        self._direct.clear()
        self._ambient.clear()
        self._streak = 0
        return removed

    def is_stale(self, job: Job, now: float) -> bool:
        return job.enqueued_at > 0 and now - job.enqueued_at > self.stale_sec

    def has_pending_from(self, chat_id: int, user_id: int | None) -> bool:
        if user_id is None:
            return False
        return any(j.user_id == user_id and j.chat_id == chat_id
                   for j in (*self._direct, *self._ambient))

    def drop_stale(self, now: float) -> int:
        """Remove jobs older than stale_sec. Returns how many were dropped."""
        dropped = 0
        for lane in (self._direct, self._ambient):
            keep = deque(j for j in lane if now - j.enqueued_at <= self.stale_sec)
            dropped += len(lane) - len(keep)
            lane.clear()
            lane.extend(keep)
        return dropped

    def pop(self, now: float) -> Job | None:
        """Pick the next job fairly. DIRECT is preferred, but after
        direct_streak_max DIRECT picks an AMBIENT one jumps the queue."""
        self.drop_stale(now)
        if self._streak >= self.direct_streak_max and self._ambient:
            self._streak = 0
            return self._ambient.popleft()
        if self._direct:
            self._streak += 1
            return self._direct.popleft()
        if self._ambient:
            self._streak = 0
            return self._ambient.popleft()
        return None
