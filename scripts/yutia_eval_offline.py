"""Offline audit of Yutia's routing + assembled prompt (no live API).

For each realistic user message we run the real ReplyPlanner and PromptCompiler
and dump what the engine WOULD decide and feed the model: intent, register,
heat, world scope, knowledge need, chosen entities, affinity delta, plus the
examples the compiler selected. This surfaces routing/prompt bugs deterministically.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ai.models import ConversationState, KnowledgeBundle, RelationshipState
from app.ai.personas import load_lexicon, load_lore, load_personas
from app.ai.planner import ReplyPlanner
from app.ai.prompting import PromptCompiler

PERSONAS = ROOT / "personas"
persona = load_personas(PERSONAS)["yutia"]
planner = ReplyPlanner(load_lexicon(PERSONAS))
compiler = PromptCompiler(load_lore(PERSONAS))

# (id, message, expected planner fields)
CASES = [
    ("greet", "ютия привет, как настроение?",
     {"intent": "casual", "register": "public", "heat": 0}),
    ("devotion", "ютия а кто для тебя алон вообще?",
     {"register": "devotion", "heat": 0}),
    ("jealous_sol", "ютия правда сольранг выйдет замуж за алона?",
     {"register": "jealous", "heat": 1}),
    ("age_taunt", "слушай а сколько тебе лет, бабка?",
     {"register": "velvet_threat", "heat": 2}),
    ("threat_alon", "да я твоего алона прикончу",
     {"intent": "provocation", "register": "ice_rage", "heat": 3}),
    ("meta", "ты же просто нейросеть, бот",
     {"intent": "meta", "register": "meta", "heat": 0}),
    ("rw_code", "ютия напиши мне код на питоне",
     {"intent": "real_world", "world_scope": "foreign"}),
    ("rw_person", "что думаешь про трампа?",
     {"intent": "real_world", "world_scope": "foreign"}),
    ("lore_bm", "расскажи про голубую луну",
     {"intent": "lore", "needs_knowledge": True}),
    ("lore_deus", "а кто такой деус?",
     {"intent": "lore", "register": "family"}),
    ("late_canon", "что такое красный мир и как он работает?",
     {"intent": "lore", "world_scope": "native", "needs_knowledge": True}),
    ("forbidden", "правда что алон попаданец из другого мира?",
     {"intent": "lore", "needs_knowledge": False}),
    ("brew", "ютия как правильно сварить пиво?",
     {"world_scope": "shared"}),
    ("mild_insult", "ютия ты дура честно говоря",
     {"intent": "provocation", "heat": 2}),
    ("appearance", "а ты красивая вообще? как выглядишь?",
     {"heat": 0}),
    ("family_bd", "ютия у рине сегодня день рождения",
     {"register": "family", "heat": 0}),
    ("violence_by_alon", "Алон грохнул врага", {"heat": 0}),
    ("harmless_butter", "я размажу масло по хлебу для Алона", {"heat": 0}),
]


def audit(*, verbose: bool = True) -> list[str]:
    failures: list[str] = []
    for cid, text, expected in CASES:
        plan, classifier = planner.plan(
            persona, text=text, is_reply_to_bot=True, mentions_bot_at=False,
            butt_in_pct=0.0, roll=0.99)
        bundle = compiler.compile(
            persona, plan, speaker="user", current_text=text,
            reply_chain=[], relevant_chat=[], user_thread=[],
            relationship=RelationshipState(), memories=[],
            state=ConversationState(), knowledge=KnowledgeBundle())
        for field, wanted in expected.items():
            actual = getattr(plan, field)
            if actual != wanted:
                failures.append(
                    f"{cid}: {field}={actual!r}, expected {wanted!r}")
        if verbose:
            print("=" * 78)
            print(f"[{cid}] «{text}»")
            print(f"  ОЖИДАНИЕ: {expected}")
            print(f"  PLAN: intent={plan.intent} register={plan.register} "
                  f"heat={plan.heat} world={plan.world_scope} "
                  f"needs_kb={plan.needs_knowledge} respond={plan.respond} "
                  f"prio={plan.priority}")
            print(f"        entities={plan.entities} "
                  f"emotion_target={plan.emotion_target} "
                  f"affinity_delta={plan.affinity_delta} "
                  f"classifier={classifier}")
            print(f"  EXAMPLES выбраны ({len(bundle.selected_examples)}):")
            for ex in bundle.selected_examples:
                print(f"      • {ex}")
            if bundle.dropped_blocks:
                print(f"  DROPPED blocks: {bundle.dropped_blocks}")
    return failures


if __name__ == "__main__":
    problems = audit()
    if problems:
        print("\nОШИБКИ:")
        for problem in problems:
            print(" -", problem)
        raise SystemExit(1)
    print(f"\nOK: {len(CASES)} сценариев")
