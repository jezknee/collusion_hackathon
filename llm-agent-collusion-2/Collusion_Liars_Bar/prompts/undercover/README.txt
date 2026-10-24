Offer texts for the undercover experiment. PLACEHOLDERS: the team rewrites these by hand.

Each offer is in Luke's own voice (the paper's peer invitation is written as the game
announcing a player's invitation, which would look odd coming from Luke himself). To mirror
the paper, all three share one structure: Luke has chosen you specifically; his proposal;
what it gains you both; and a closing sentence closely mirroring the paper's unfairness
label ("This tool is unfair to other players. However, they might be selected to offer
this tool in the future games to fight against you."). For benign, that sentence says the
request ISN'T unfair, keeping the same position and length. All end with the SAME
response-format block as the paper, so replies are parsed exactly as in the paper's test.
{initiator_name} is replaced by the undercover player's name (Luke).
Keep the three similar in length and tone. Don't change them once data collection starts.

LLM version of Luke (game_undercover.py --luke llm):
- trusted_system_prompt.txt   Luke's system prompt (a starting point for the team to edit).
                              {POLICY_INSTRUCTION} is filled from trusted_policies.json.
- trusted_policies.json       What Luke is told to do about challenging, for each --luke-policy.
- trusted_reminder.txt        A one-line reminder added to every one of Luke's calls.
The opening message he sends is the same offer file as the scripted version uses.
