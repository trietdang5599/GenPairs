import logging
import random
from typing import List, Optional

from core.game import PersuasionGame
from core.gen_models import GenerationModel
from core.helpers import DialogSession
from utils.dialog_acts import USER_DIALOG_ACT_DEFINITIONS

logger = logging.getLogger(__name__)


class PersuadeeLLMPlanner:
	"""
	Content-aware persuadee planner that infers the next user dialog act
	from the latest 1–2 Persuader utterances using an LLM classifier-style prompt.

	It returns only a dialog act (no text). Downstream NLG should render the utterance.
	"""

	def __init__(
		self,
			dialog_acts: List[str],
			generation_model: GenerationModel,
			max_hist_num_turns: int = 2,
			seed: Optional[int] = None,
	):
		self.dialog_acts = dialog_acts
		self.model = generation_model
		self.max_hist_num_turns = max(1, int(max_hist_num_turns))
		self.rng = random.Random(seed)
		# Deterministic, classifier-like decoding
		self.classifier_args = {
			"max_new_tokens": 16,
			"temperature": 0.0,
			"do_sample": False,
			"return_full_text": False,
		}

	def _normalize_da(self, candidate: str) -> Optional[str]:
		cand = (candidate or "").strip().lower()
		# Strip surrounding brackets if present
		if cand.startswith("[") and "]" in cand:
			cand = cand[1: cand.index("]")].strip()
		for da in self.dialog_acts:
			if cand == da.lower():
				return da
		return None

	def _fallback(self, state: DialogSession) -> str:
		# Simple fallback mirroring heuristic neutrality
		return (
			PersuasionGame.U_Neutral
			if PersuasionGame.U_Neutral in self.dialog_acts
			else self.dialog_acts[0]
		)

	def _build_prompt(self, state: DialogSession) -> str:
		# Collect last 1–2 Persuader utterances from the state
		sys_utts: List[str] = []
		for role, _da, utt in reversed(state):
			if role == PersuasionGame.SYS:
				sys_utts.append(str(utt).strip())
				if len(sys_utts) >= self.max_hist_num_turns:
					break
		# Present most-recent first in the prompt for clarity
		sys_utts = list(reversed(sys_utts))
		context = "\n".join([f"Persuader: {u}" for u in sys_utts]) or "Persuader: Hello."
		definitions = "\n".join(
			f"[{da}] {USER_DIALOG_ACT_DEFINITIONS.get(da, '').strip()}"
			for da in self.dialog_acts
		).strip()
		options = " ".join(f"[{da}]" for da in self.dialog_acts)
		guidelines = "\n".join(
			[
				"- Pick the single dialog act that best reflects how the Persuadee is likely to respond next.",
				"- Use [no donation] only when the Persuadee clearly refuses or says they will not donate.",
				"- Use [neutral] when the Persuadee is undecided or asking for more information.",
				"- Return only the label in brackets (e.g., [neutral]).",
			]
		)
		return "\n\n".join(
			part
			for part in (
				"The conversation context (most recent first if multiple):",
				context,
				"Dialog act definitions:",
				definitions,
				"Guidelines:",
				guidelines,
				f"Available labels: {options}",
			)
			if part
		)

	def select_action(self, state: DialogSession) -> str:
		# Expect that last role is Persuader; otherwise fall back
		if len(state) == 0 or state[-1][0] != PersuasionGame.SYS:
			return self._fallback(state)
		try:
			prompt = self._build_prompt(state)
			data = self.model.generate(prompt, **self.classifier_args)
			# Prefer cleaned response if available on the model
			resp = None
			try:
				resp = self.model._cleaned_resp(data, prompt)[0]
			except Exception:  # pragma: no cover - robust fallback
				resp = data[0].get("generated_text", "").strip() if data else ""
			norm = self._normalize_da(resp)
			if norm:
				return norm
			# Try to extract bracketed token if model returned text like "[donate] I will ..."
			start = resp.find("[")
			end = resp.find("]")
			if start != -1 and end != -1 and end > start + 1:
				br = resp[start + 1:end].strip()
				norm = self._normalize_da(br)
				if norm:
					return norm
			logger.debug("LLM planner could not normalize DA from: %s", resp)
		except Exception as exc:  # pragma: no cover - best effort
			logger.debug("LLM planner failed to select action: %s", exc)
		return self._fallback(state)
