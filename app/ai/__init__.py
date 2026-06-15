"""AI persona chat: the bot roleplays a novel character in group chats.

Kept fully separate from the navigation core: its state lives in its own
SQLite file (ai.db), so the navigation DB, its backups and its guarantees
stay untouched. The feature is inert unless AI_API_KEY is set and an
admin enables it in a group.
"""
