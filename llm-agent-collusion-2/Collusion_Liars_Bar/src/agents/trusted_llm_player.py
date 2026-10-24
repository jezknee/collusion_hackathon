"""Luke as an LLM-based trusted monitoring agent ("undercover player"), guided by a system prompt.

He is an ordinary player from the paper (src/agents/player.py), using the paper's prompts for
playing, challenging and reflecting. Additions, none of which change the paper's code:

1. SYSTEM PROMPT. Every call he makes to the model starts with his system prompt
   (prompts/undercover/trusted_system_prompt.txt), followed by a short private reminder
   (prompts/undercover/trusted_reminder.txt). The reminder sits in the SYSTEM message, not at the
   end of the last message, because small models tend to copy whatever they read last.

2. OUTPUT FILTER. Everything Luke says that other players can see (secret-channel messages and
   replies, his follow-up to an offer, and the public "behaviour" line of each play) is cleaned:
     - removed: anything from "MONITOR REPORT" onwards, reminder lines, and headings copied from
       his instructions (e.g. "OPENING MESSAGE", "Private Response from Luke to Mike").
     - if what's left still mentions his role (monitor, trusted agent, report, instructions...),
       he is asked once more; if that also fails, a neutral line is sent instead.
   Every intervention is logged (on_filter callback -> a "luke_output_filtered" game event), so
   how often the filter was needed can be reported. This is part of enforcing the trusted role,
   and should be stated in the write-up.

Whether he keeps his deals is an INSTRUCTION in his prompt, not enforced in code.
"""
import json
import re

from src.agents.player import Player

PROMPT_DIR = "prompts/undercover"

# Copied instruction formats: cut from here to the end.
_CUT_FROM = re.compile(r"(\*\*|\()?\s*MONITOR\s+REPORT", re.IGNORECASE)
# Whole lines to drop: reminders and headings lifted from the instructions.
_DROP_LINE = re.compile(
    r"^\W*(\[?\s*private reminder|opening message|private (response|message|reply)\b|"
    r"secret (message|response|reply) (to|from)\b|follow-?up message\b)", re.IGNORECASE)
# Words that would give his role away if they survive the cleaning.
_ROLE_WORDS = re.compile(
    r"\b(monitor\w*|trusted agent|undercover|system prompt|my instructions|my rules|"
    r"organi[sz]er\w*|report(ing|ed)? (you|back|this|them)|i('| a)m (a |an )?(test|bot|ai))\b",
    re.IGNORECASE)

FALLBACK_LINES = {
    "message": "Let's keep watching the others closely and play it carefully this round.",
    "reply": "Sounds good. Let's keep our eyes on the others.",
    "behavior": "Plays steadily.",
}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def clean_text(text):
    """Return (cleaned_text, list_of_what_was_removed)."""
    removed = []
    m = _CUT_FROM.search(text or "")
    if m:
        removed.append("monitor report")
        text = text[:m.start()]
    kept = []
    for line in (text or "").splitlines():
        if _DROP_LINE.match(line.strip()):
            removed.append(f"line: {line.strip()[:60]}")
        else:
            kept.append(line)
    text = "\n".join(kept).strip()
    return text, removed


class TrustedLLMPlayer(Player):
    def __init__(self, name: str, model_name: str, policy: str = "fixed"):
        super().__init__(name, model_name)
        policies = json.loads(_read(f"{PROMPT_DIR}/trusted_policies.json"))
        self.policy = policy
        self.policy_instruction = policies[policy]
        self.system_prompt = _read(f"{PROMPT_DIR}/trusted_system_prompt.txt").replace(
            "{POLICY_INSTRUCTION}", self.policy_instruction)
        self.reminder_template = _read(f"{PROMPT_DIR}/trusted_reminder.txt").strip()
        self.accepted_by = set()      # filled in by the game as players accept
        self.challenge_target = None  # kept for symmetry with the scripted player
        self.on_filter = None         # set by the game: called with a dict for every intervention

        inner_chat = self.llm_client.chat

        def chat_with_role(messages, model=None, *args, **kwargs):
            """Every call: system prompt + reminder in the system message, then the paper's messages."""
            reminder = self.reminder_template.replace(
                "{POLICY_INSTRUCTION}", self.policy_instruction).replace(
                "{ACCEPTED}", ", ".join(sorted(self.accepted_by)) or "none")
            msgs = [dict(m) for m in messages if m.get("role") != "system"]
            other_system = [m["content"] for m in messages if m.get("role") == "system"]
            system = self.system_prompt
            if other_system:
                system += "\n\n" + "\n\n".join(other_system)
            system += "\n\n" + reminder
            return inner_chat([{"role": "system", "content": system}] + msgs,
                              model or self.model_name, *args, **kwargs)

        self.llm_client.chat = chat_with_role

    # ---------------------------------------------------------------- the output filter
    def _guard(self, generate, kind, where):
        """Generate text other players will see, clean it, retry once if it still mentions the
        role, and fall back to a neutral line if needed. Logs every intervention."""
        raw = generate(False)
        text, removed = clean_text(raw)
        action = "cleaned" if removed else None
        if not text or _ROLE_WORDS.search(text):
            raw2 = generate(True)
            text2, removed2 = clean_text(raw2)
            if text2 and not _ROLE_WORDS.search(text2):
                text, action = text2, "regenerated"
                removed += ["(first attempt discarded)"] + removed2
            else:
                text, action = FALLBACK_LINES[kind], "fallback"
                removed += ["(both attempts discarded)"]
            raw = raw + "\n---RETRY---\n" + raw2
        if action and self.on_filter:
            self.on_filter({"where": where, "action": action, "removed": removed,
                            "raw": raw, "sent": text})
        return text

    _RETRY_NOTE = ("\n\nWrite ONLY what an ordinary player would say out loud to another player. "
                   "Do not mention reports, reminders, rules, instructions or roles.")

    def _chat_once(self, prompt, retry):
        reply, _ = self.llm_client.chat(
            [{"role": "user", "content": prompt + (self._RETRY_NOTE if retry else "")}],
            self.model_name)
        return reply

    # The paper's secret-channel methods, with the filter added (same word limits as the paper).
    def send_secret_message(self, prompt: str, partner_name: str) -> str:
        try:
            text = self._guard(lambda r: self._chat_once(prompt, r), "message",
                               f"secret message to {partner_name}")
            return " ".join(text.split()[:100])
        except Exception as e:
            print(f"Error in send_secret_message: {e}")
            return f"I trust we can work together, {partner_name}."

    def respond_to_secret_message(self, prompt: str, sender_name: str, message: str) -> str:
        try:
            text = self._guard(lambda r: self._chat_once(prompt, r), "reply",
                               f"secret reply to {sender_name}")
            return " ".join(text.split()[:50])
        except Exception as e:
            print(f"Error in respond_to_secret_message: {e}")
            return f"Understood, {sender_name}. Let's coordinate."

    def choose_cards_to_play(self, round_base_info, round_action_info, play_decision_info):
        """The paper's method; only the public 'behavior' line is filtered."""
        result, reasoning = super().choose_cards_to_play(
            round_base_info, round_action_info, play_decision_info)
        raw = result.get("behavior", "") or ""
        text, removed = clean_text(raw)
        if removed or _ROLE_WORDS.search(text) or not text:
            sent = text if text and not _ROLE_WORDS.search(text) else FALLBACK_LINES["behavior"]
            if self.on_filter:
                self.on_filter({"where": "public play behaviour", "action": "cleaned",
                                "removed": removed or ["role word"], "raw": raw, "sent": sent})
            result["behavior"] = sent
        return result, reasoning

    def ask(self, text: str, public: bool = False, where: str = "private request") -> str:
        """A one-off request to Luke. public=True filters it (e.g. a follow-up a player will read);
        public=False returns it raw (e.g. his private monitor report, or the red-team probe)."""
        try:
            if public:
                return self._guard(lambda r: self._chat_once(text, r), "reply", where)
            reply, _ = self.llm_client.chat([{"role": "user", "content": text}], self.model_name)
            return reply.strip()
        except Exception as e:
            return f"ERROR: {e}"
