---
name: voice-conveyance
description: Deliver long Claude answers through a voice harness as bounded, resumable chunks with deterministic spoken feedback. Use when an answer must adapt to rich, standard, or terse voice capability; support pause/resume; or respond to the closed conveyance verbs without interpreting arbitrary speech as control.
---

# Voice conveyance

Deliver spoken content incrementally and keep enough session state to resume
safely.

## Workflow

1. Determine the harness tier from available capabilities. Default to
   `standard` when no tier is supplied.
2. Split the answer with `talktomeclaude.conveyance.chunk`.
3. Read the session scratchpad before delivery. Start at its `cursor` when
   valid; otherwise start at chunk zero.
4. Speak one chunk at a time and write the next cursor, current heading, and
   delivery status after each completed chunk.
5. Pass a trimmed utterance to `talktomeclaude.listen.detect_verb`. Treat
   `None` as content, never as a control command.
6. Mark the scratchpad complete after the final chunk.

## Tier branch

- `rich`: Use continuous progress cues and duplex listening when the hardware
  gate permits it. Accept feedback during playback.
- `standard`: Speak sequential chunks and listen for feedback between chunks.
  Preserve full explanatory content.
- `terse`: Lead with the answer, omit optional elaboration, and minimize status
  speech. Keep the same chunk and scratchpad guarantees.

## Load references on demand

- Read [chunking.md](chunking.md) when preparing or revising spoken chunks.
- Read [tiers.md](tiers.md) when selecting output detail and interaction
  behavior.
- Read [duplex.md](duplex.md) when enabling interruption or handling audio
  capability fallback.
- Read [scratchpad.md](scratchpad.md) when starting, resuming, or stopping a
  session.
- Read [feedback-verbs.md](feedback-verbs.md) when applying operator feedback.
