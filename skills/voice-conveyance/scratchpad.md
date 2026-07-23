# Resume scratchpad

Store session delivery state at
`.omc/state/sessions/{session_id}/voice-conveyance.json` through the conveyance
helpers. The only fields are:

- `cursor`: zero-based index of the next chunk to speak.
- `heading`: the answer section currently being delivered.
- `status`: delivery state such as `delivering`, `stopped`, or `complete`.

Read before resuming. Write after a chunk finishes and whenever delivery stops
or completes. Treat `{}` as no resumable state; do not reconstruct state from a
missing, unreadable, or invalid file.
