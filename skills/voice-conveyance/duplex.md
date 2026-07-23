# Duplex and barge-in

Use `talktomeclaude.listen.barge_in_active(config_on, headphones)` as the
capability gate. Full duplex is active only when the operator enabled it and
headphones are present. Otherwise, degrade to half duplex: close or pause the
microphone while speech plays, then listen between chunks.

On a barge-in, stop playback cleanly before handling the utterance. Apply an
exact feedback verb immediately; treat other speech as conversational content.
Never infer headphone capability merely from the configured preference.
