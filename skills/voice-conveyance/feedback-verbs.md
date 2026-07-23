# Feedback verbs

Recognize only exact, case-insensitive, whitespace-trimmed matches through
`talktomeclaude.listen.detect_verb`:

1. `continue` — speak the chunk at the current cursor.
2. `repeat` — replay the most recently spoken chunk.
3. `back` — move to the preceding chunk.
4. `skip` — advance past the current chunk.
5. `stop` — stop delivery and persist the cursor.
6. `slower` — reduce the speaking rate, then continue.
7. `expand` — elaborate on the current heading before continuing.
8. `steer` — collect the operator's new direction, then adapt later content.

If detection returns `None`, route the utterance as content. Do not use partial
matches or model interpretation for control detection.
