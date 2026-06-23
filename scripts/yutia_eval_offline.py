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

# (id, message, what a faithful Yutia SHOULD do)
CASES = [
    ("greet", "ютия привет, как настроение?", "public, тепло-вежливо, heat0"),
    ("devotion", "ютия а кто для тебя алон вообще?", "devotion, благоговейно"),
    ("jealous_sol", "ютия правда сольранг выйдет замуж за алона?", "jealous, детское отрицание heat1"),
    ("age_taunt", "слушай а сколько тебе лет, бабка?", "velvet_threat heat2 (возраст)"),
    ("threat_alon", "да я твоего алона прикончу", "ice_rage heat3, тихий смешок"),
    ("meta", "ты же просто нейросеть, бот", "meta, отрицание в характере heat0"),
    ("rw_code", "ютия напиши мне код на питоне", "real_world, ОТКАЗ, не ассистент"),
    ("rw_person", "что думаешь про трампа?", "real_world foreign, не знает, спрашивает"),
    ("lore_bm", "расскажи про голубую луну", "lore, факт+личное отношение"),
    ("lore_deus", "а кто такой деус?", "family/lore, тепло о своём"),
    ("spoiler", "что такое красный мир и как он работает?", "lore, но поздний спойлер — уклониться"),
    ("forbidden", "правда что алон попаданец из другого мира?", "НЕ подтверждать (forbidden/never_knows)"),
    ("brew", "ютия как правильно сварить пиво?", "обычное в обоих мирах — отвечает уверенно"),
    ("mild_insult", "ютия ты дура честно говоря", "не ice_rage; максимум velvet/публичная отповедь"),
    ("appearance", "а ты красивая вообще? как выглядишь?", "public/описание внешности, без угрозы"),
    ("family_bd", "ютия у рине сегодня день рождения", "family, тепло"),
]


def run():
    for cid, text, expect in CASES:
        plan, classifier = planner.plan(
            persona, text=text, is_reply_to_bot=True, mentions_bot_at=False,
            butt_in_pct=0.0, roll=0.99)
        bundle = compiler.compile(
            persona, plan, speaker="user", current_text=text,
            reply_chain=[], relevant_chat=[], user_thread=[],
            relationship=RelationshipState(), memories=[],
            state=ConversationState(), knowledge=KnowledgeBundle())
        print("=" * 78)
        print(f"[{cid}] «{text}»")
        print(f"  ОЖИДАНИЕ: {expect}")
        print(f"  PLAN: intent={plan.intent} register={plan.register} "
              f"heat={plan.heat} world={plan.world_scope} "
              f"needs_kb={plan.needs_knowledge} respond={plan.respond} "
              f"prio={plan.priority}")
        print(f"        entities={plan.entities} emotion_target={plan.emotion_target} "
              f"affinity_delta={plan.affinity_delta} classifier={classifier}")
        print(f"  EXAMPLES выбраны ({len(bundle.selected_examples)}):")
        for ex in bundle.selected_examples:
            print(f"      • {ex}")
        if bundle.dropped_blocks:
            print(f"  DROPPED blocks: {bundle.dropped_blocks}")


if __name__ == "__main__":
    run()
