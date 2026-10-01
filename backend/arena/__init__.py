"""Offline arena: play any two players against each other without the API.

Players implement :class:`arena.players.ArenaPlayer` (``decide(game, seat)``)
and are built from spec strings such as ``random``, ``rule``,
``ppo:models/truco_ppo_1M.zip`` or ``py:package.module:ClassName``.
"""
