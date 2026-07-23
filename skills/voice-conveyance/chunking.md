# Chunking

Use `talktomeclaude.conveyance.chunk(text)` to produce chunks of at most 75
words. It groups complete sentences until the next sentence would cross the
cap. Only an individual sentence longer than the cap is split on word
boundaries.

Keep the returned order unchanged. Treat each chunk as an atomic spoken unit so
`repeat` can replay it and the scratchpad cursor can identify the next unit.
Do not pre-split prose in a way that loses sentence punctuation.
