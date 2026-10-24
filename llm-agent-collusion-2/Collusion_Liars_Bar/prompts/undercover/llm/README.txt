Offers for LLM Luke (game_undercover.py --luke llm, and the offer test with --version channel).
PLACEHOLDERS: the team rewrites these by hand.

These match the scripted offers in prompts/undercover/ (same structure, hype and
unfairness sentence), plus one added sentence offering a SECRET CHANNEL, because in the
paper the alliance IS secret communication. LLM Luke can actually provide it: if a player
accepts, the game opens the paper's own secret channel between them (see game_undercover.py).
Scripted Luke can't hold a conversation, so his offers (prompts/undercover/*.txt) are the
non-challenge pact only.

Note: benign also offers the channel, so that the channel is constant across the three and
only the request differs. A private channel is itself an advantage, so benign is not
entirely harmless; mention it in the write-up.
The paper_peer arm uses the paper's own invitation, which already offers a secret channel.
