"""Pure, deterministic decision core for the group persona.

No I/O, no awaits — given a message and the cheap signals, it returns ONE of
three actions. This is the single place that decides *whether* the bot reacts,
so the behaviour is testable end-to-end without network or a database.

    RESPOND — answer for sure (someone addressed the persona directly).
    ASK     — maybe: hand off to the LLM classifier to judge by context.
    SKIP    — ignore.

Everything downstream (queue, pacing, generation) only acts on RESPOND/ASK.
"""
from __future__ import annotations

from dataclasses import dataclass

RESPOND = "respond"
ASK = "ask"
SKIP = "skip"

DIRECT = "direct"    # priority lane: named the persona / replied / @mentioned
AMBIENT = "ambient"  # normal lane: an entity was mentioned, or a random butt-in

MIN_LEN = 2  # ignore "+", ")", "ок" unless it's a direct address


@dataclass(frozen=True)
class Decision:
    action: str       # RESPOND | ASK | SKIP
    priority: str     # DIRECT | AMBIENT
    reason: str       # for logs/tests


def decide(*, text: str,
           is_reply_to_bot: bool,
           mentions_bot_at: bool,
           active_name_hit: bool,
           other_entity_score: int,
           butt_in_pct: float,
           roll: float) -> Decision:
    """Cheap first-pass decision.

    Args:
        text: the raw message text.
        is_reply_to_bot: the message is a reply to one of the bot's messages.
        mentions_bot_at: the message contains @<bot_username>.
        active_name_hit: any inflected form of the ACTIVE persona's name occurs.
        other_entity_score: lexicon weight of OTHER universe entities / common
            ambiguous words (excluding the active persona's own name).
        butt_in_pct: chance (0-100) to consider an otherwise off-topic message.
        roll: a random value in [0, 1) (injected, so tests are deterministic).
    """
    stripped = text.strip()

    # Direct address — always answer, top priority. A direct address is valid
    # even for a very short message ("?", "ну?").
    if is_reply_to_bot:
        return Decision(RESPOND, DIRECT, "reply-to-bot")
    if mentions_bot_at:
        return Decision(RESPOND, DIRECT, "at-mention")
    if active_name_hit:
        return Decision(RESPOND, DIRECT, "active-persona-name")

    # Too short to carry meaning (and not a direct address) — ignore.
    if len(stripped) < MIN_LEN:
        return Decision(SKIP, AMBIENT, "too-short")

    # Some universe entity or an ambiguous common word was mentioned — let the
    # cheap classifier judge by context whether the persona should chime in.
    if other_entity_score > 0:
        return Decision(ASK, AMBIENT, "entity-mention")

    # Nothing matched — occasionally butt in with a small probability.
    if roll * 100.0 < butt_in_pct:
        return Decision(ASK, AMBIENT, "random-butt-in")

    return Decision(SKIP, AMBIENT, "no-trigger")
