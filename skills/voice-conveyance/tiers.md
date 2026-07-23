# Harness tiers

- **rich**: Audio output plus live input/progress are available. Permit
  capability-gated barge-in and provide brief progress cues.
- **standard**: Audio output and turn-by-turn input are available. Speak one
  chunk, then collect feedback before continuing.
- **terse**: The harness has limited interaction or output budget. State the
  answer first, remove optional examples, and keep transitions minimal.

All tiers retain the 75-word cap, exact feedback matching, and scratchpad
updates. If the tier is unknown, use `standard`.
