import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
try:
	import torch
except ImportError:
	torch = None
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
from core.helpers import DialogSession

if TYPE_CHECKING:
	from core.game import PersuasionGame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_LOG_PATH = PROJECT_ROOT / "logs" / "prompt_log.txt"
_P4G_DIALOG_CACHE: Dict[str, List[Tuple[str, dict]]] = {}
_P4G_ANCHOR_POINTERS: Dict[str, int] = {}


def log_prompt(prompt: str, log_path: Optional[Path] = None) -> None:
	text = (prompt or "").strip()
	if not text:
		return
	output_path = Path(log_path) if log_path else PROMPT_LOG_PATH
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if not text.endswith("\n"):
		text = f"{text}\n"
	with output_path.open("a", encoding="utf-8") as handle:
		handle.write("---------\n")
		handle.write(text)
		handle.write("---------\n")


def format_messages_for_log(messages: List[Dict[str, Any]]) -> str:
	lines: List[str] = []
	for message in messages:
		role = message.get("role", "unknown")
		content = message.get("content", "")
		if isinstance(content, list):
			parts: List[str] = []
			for chunk in content:
				if isinstance(chunk, dict):
					text = chunk.get("text")
					if text is None:
						text = str({k: v for k, v in chunk.items() if k != "text"})
					parts.append(str(text))
				else:
					parts.append(str(chunk))
			content_str = "\n".join(parts)
		else:
			content_str = str(content)
		lines.append(f"{role}: {content_str}".strip())
	return "\n".join(lines)

def set_determinitic_seed(seed: Optional[int], enforce_determinism: bool = False) -> None:
	"""
	Seed Python, NumPy, and PyTorch RNGs. When `seed` is None we skip reseeding so runs remain stochastic.
	If `enforce_determinism` is True we attempt to enable deterministic algorithms (falling back gracefully
	if the current PyTorch version lacks support).
	"""
	if seed is None:
		if enforce_determinism:
			logging.getLogger(__name__).warning(
				"Deterministic mode requested but no seed provided; leaving RNG state unchanged."
			)
		return None

	if torch is None:
		if enforce_determinism:
			logging.getLogger(__name__).warning("PyTorch is not installed; skipping torch-specific seeding.")
		random.seed(seed)
		np.random.seed(seed)
		return None

	if enforce_determinism and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
		os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	try:
		if enforce_determinism:
			torch.use_deterministic_algorithms(True, warn_only=True)
		else:
			torch.use_deterministic_algorithms(False)
	except TypeError:
		# Older PyTorch versions do not accept warn_only.
		torch.use_deterministic_algorithms(enforce_determinism)

	torch.backends.cudnn.deterministic = enforce_determinism
	torch.backends.cudnn.benchmark = not enforce_determinism
	return None


class dotdict(dict):
	def __getattr__(self, name):
		return self[name]


class hashabledict(dict):
	def __hash__(self):
		return hash(tuple(sorted(self.items())))


logger = logging.getLogger(__name__)


def get_preference_pair(
	probabilities: np.ndarray,
	state_rep: str,
	dialog_acts: Iterable[str],
	valid_moves: Iterable[int],
	realizations_vs: Optional[Dict[str, Dict[str, float]]],
) -> Optional[Tuple[int, Tuple[str, float], Tuple[str, float]]]:
	if not realizations_vs:
		return None

	probabilities = np.asarray(probabilities)
	if probabilities.size == 0:
		return None

	valid_moves_list = [int(action_idx) for action_idx in valid_moves]
	if not valid_moves_list:
		return None

	best_prob = -float("inf")
	target_idx = None
	for action_idx in valid_moves_list:
		prob_val = float(probabilities[action_idx])
		if prob_val > best_prob:
			best_prob = prob_val
			target_idx = action_idx

	if target_idx is None:
		return None

	dialog_acts_list = list(dialog_acts)
	if 0 <= target_idx < len(dialog_acts_list):
		label = dialog_acts_list[target_idx]
	else:
		label = str(target_idx)

	prefetch_key = f"{state_rep}__{label}"
	realization_dict = realizations_vs.get(prefetch_key)
	if not realization_dict or len(realization_dict) < 2:
		return None

	best_pair = max(realization_dict.items(), key=lambda kv: kv[1])
	worst_pair = min(realization_dict.items(), key=lambda kv: kv[1])

	return target_idx, best_pair, worst_pair


def summarize_action_statistics(
	probabilities: np.ndarray,
	state_rep: str,
	dialog_acts: Iterable[str],
	valid_moves: Iterable[int],
	q_table: Dict[str, Dict[int, float]],
	nsa_table: Dict[str, Dict[int, int]],
	realizations_vs: Optional[Dict[str, Dict[str, float]]] = None,
) -> tuple[str, str]:
	"""Build human-readable summaries for the action distribution explored by MCTS.

	Args:
		probabilities: Array of action probabilities (aligned with dialog_acts).
		state_rep: String representation of the current state.
		dialog_acts: Ordered iterable mapping action index -> dialog act label.
		valid_moves: Iterable of action indices that were considered valid.
		q_table: Mapping state -> {action: Q-value}.
		nsa_table: Mapping state -> {action: visit count}.
		realizations_vs: Optional mapping for OpenLoopMCTS storing realizations and their values.

	Returns:
		Tuple of (header_line, detail_block) suitable for logging.
	"""
	probabilities = np.asarray(probabilities)
	dialog_acts = list(dialog_acts)
	valid_moves = list(valid_moves)
	best_idx = int(np.argmax(probabilities)) if probabilities.size else None

	def _label_for(action_idx: int) -> str:
		if 0 <= action_idx < len(dialog_acts):
			return dialog_acts[action_idx]
		return str(action_idx)

	best_label = _label_for(best_idx) if best_idx is not None else "None"
	best_prob = float(probabilities[best_idx]) if best_idx is not None else None
	state_qs = q_table.get(state_rep, {})
	state_nsas = nsa_table.get(state_rep, {})
	best_q = state_qs.get(best_idx) if best_idx is not None else None
	best_visits = state_nsas.get(best_idx) if best_idx is not None else None
	best_prob_str = f"{best_prob:.3f}" if best_prob is not None else "None"
	best_q_str = f"{best_q:.4f}" if best_q is not None else "None"
	best_visits_str = str(best_visits) if best_visits is not None else "None"

	header = (
		f"Visit distribution: {[float(f'{p:.3f}') for p in probabilities]}; "
		f"best action={best_idx} ({best_label}) (prob={best_prob_str}, Q={best_q_str}, visits={best_visits_str})"
	)

	strategy_lines = []
	for action_idx in valid_moves:
		label = _label_for(action_idx)
		prob_val = float(probabilities[action_idx])
		sentence = "N/A"
		realization_info = []
		if realizations_vs:
			prefetch_key = f"{state_rep}__{label}"
			realization_dict = realizations_vs.get(prefetch_key)
			if realization_dict:
				best_utt, _ = max(realization_dict.items(), key=lambda kv: kv[1])
				sentence = best_utt
				realization_info = [
					f"- {utt} (v={val:.3f})" for utt, val in realization_dict.items()
				]
		entry = f"[{label} | {sentence} | {prob_val:.3f}]"
		if realization_info:
			entry += "\n\t" + "\n\t".join(realization_info)
		strategy_lines.append(entry)

	detail_block = "\n".join(strategy_lines)
	return header, detail_block



def load_p4g_dialogs(dataset_path: Optional[Path] = None) -> List[Tuple[str, dict]]:
	"""Stream and cache the P4G train dialogs from disk."""
	if dataset_path is None:
		resolved_path = (PROJECT_ROOT / "data" / "p4g" / "300_dialog_turn_based-train.jsonl").resolve()
	else:
		resolved_path = dataset_path.resolve()
	key = str(resolved_path)
	if key in _P4G_DIALOG_CACHE:
		return _P4G_DIALOG_CACHE[key]
	if not resolved_path.exists():
		logger.warning("P4G dataset not found at %s – simulations will start empty.", resolved_path)
		_P4G_DIALOG_CACHE[key] = []
		return _P4G_DIALOG_CACHE[key]

	items: List[Tuple[str, dict]] = []
	try:
		with resolved_path.open("r", encoding="utf-8") as handle:
			for line in handle:
				line = line.strip()
				if not line:
					continue
				try:
					record = json.loads(line)
				except json.JSONDecodeError:
					logger.debug("Skipping malformed P4G dialog line.")
					continue
				dialog = record.get("dialog")
				label = record.get("label")
				if not isinstance(dialog, list) or not isinstance(label, list):
					continue
				dialog_id = record.get("id") or f"p4g_{len(items):05d}"
				items.append((dialog_id, {"dialog": dialog, "label": label}))
	except Exception as exc:
		logger.warning("Failed to load P4G dataset from %s: %s", resolved_path, exc)
		items = []

	if not items:
		logger.warning("P4G dataset at %s is empty.", resolved_path)

	_P4G_DIALOG_CACHE[key] = items
	return _P4G_DIALOG_CACHE[key]


def get_anchor_progress(dataset_path: Optional[Path] = None) -> Optional[dict[str, object]]:
	"""Return the next anchor index and dialog metadata for the given dataset.

	Args:
		dataset_path: Optional dataset override. When None the default P4G path is used.

	Returns:
		Dictionary with keys `dataset_path`, `next_index`, `total_dialogs`, and `next_dialog_id`.
		Returns None if the dataset cannot be loaded.
	"""
	dialog_items = load_p4g_dialogs(dataset_path=dataset_path)
	if not dialog_items:
		return None
	if dataset_path is not None:
		resolved_dataset = dataset_path.resolve()
	else:
		resolved_dataset = (PROJECT_ROOT / "data" / "p4g" / "300_dialog_turn_based-train.jsonl").resolve()
	key = str(resolved_dataset)
	total = len(dialog_items)
	next_idx = _P4G_DIALOG_CACHE.get(key, 0) % total
	next_dialog_id = dialog_items[next_idx][0]
	return {
		"dataset_path": str(resolved_dataset),
		"next_index": next_idx,
		"total_dialogs": total,
		"next_dialog_id": next_dialog_id,
	}


def map_system_dialog_act(raw_das, system_dialog_acts: Iterable[str]) -> str:
	"""Map raw dialog-acts onto the system ontology, defaulting to 'other'."""
	system_dialog_acts = list(system_dialog_acts)
	if not system_dialog_acts:
		return "other"

	if not raw_das:
		return system_dialog_acts[0]

	if isinstance(raw_das, str):
		return raw_das if raw_das in system_dialog_acts else system_dialog_acts[0]

	if isinstance(raw_das, (list, tuple, set)):
		for candidate in reversed(list(raw_das)):
			if candidate in system_dialog_acts:
				return candidate
		if "other" in system_dialog_acts:
			return "other"

	return system_dialog_acts[0]


def seed_with_p4g_anchor(
	dataset_path: Optional[Path],
	state: DialogSession,
	game: "PersuasionGame",
	conversation: List[dict],
	max_pairs: int = 1,
	dialogs: Optional[List[Tuple[str, dict]]] = None,
) -> int:
	"""Bootstrap a dialog with turns sampled from the P4G train set."""
	dialog_items = dialogs if dialogs is not None else load_p4g_dialogs(dataset_path=dataset_path)
	if not dialog_items:
		return 0
	if dataset_path is None:
		resolved_path = (PROJECT_ROOT / "data" / "p4g" / "300_dialog_turn_based-train.jsonl").resolve()
	else:
		resolved_path = dataset_path.resolve()
	key = str(resolved_path)
	total_dialogs = len(dialog_items)
	start_idx = _P4G_ANCHOR_POINTERS.get(key, 0) % total_dialogs

	for offset in range(total_dialogs):
		dialog_idx = (start_idx + offset) % total_dialogs
		dialog_id, dialog_entry = dialog_items[dialog_idx]
		dialog_turns = dialog_entry.get("dialog") or []
		label_turns = dialog_entry.get("label") or []
		if not dialog_turns or not label_turns:
			continue

		pairs_available = min(len(dialog_turns), len(label_turns), max_pairs)
		if pairs_available <= 0:
			continue

		seeded = 0
		for pair_idx in range(pairs_available):
			turn = dialog_turns[pair_idx]
			if not isinstance(turn, dict):
				continue
			label_entry = label_turns[pair_idx] if pair_idx < len(label_turns) else {}

			sys_tokens = turn.get("er") if isinstance(turn.get("er"), list) else []
			usr_tokens = turn.get("ee") if isinstance(turn.get("ee"), list) else []
			sys_utt_raw = " ".join(sys_tokens).strip()
			usr_utt_raw = " ".join(usr_tokens).strip()
			if not sys_utt_raw and not usr_utt_raw:
				continue

			sys_da = map_system_dialog_act(label_entry.get("er"), game.system_agent.dialog_acts)
			raw_usr_labels = label_entry.get("ee")
			if isinstance(raw_usr_labels, list) and raw_usr_labels:
				raw_usr_da = raw_usr_labels[-1]
			elif isinstance(raw_usr_labels, str):
				raw_usr_da = raw_usr_labels
			else:
				raw_usr_da = game.U_Neutral
			usr_da = game.map_user_da(raw_usr_da)

			sys_utt = sys_utt_raw or "..."
			usr_utt = usr_utt_raw or "..."
			state.add_single(game.SYS, sys_da, sys_utt)
			state.add_single(game.USR, usr_da, usr_utt)

			conversation.append(
				{
					"turn": len(conversation),
					"action_index": None,
					"system_dialog_act": sys_da,
					"system_utterance": sys_utt,
					"user_selected_act": None,
					"user_dialog_act": usr_da,
					"user_utterance": usr_utt,
					"turn_type": "anchor",
					"anchor_dialog_id": dialog_id,
				}
			)
			seeded += 1

		if seeded > 0:
			setattr(state, "_anchor_dialog_id", dialog_id)
			_P4G_ANCHOR_POINTERS[key] = (dialog_idx + 1) % total_dialogs
			return seeded

	logger.warning("Unable to sample anchor turns from P4G dataset; proceeding without anchors.")
	_P4G_ANCHOR_POINTERS[key] = (start_idx + 1) % total_dialogs
	return 0


def export_preference_pair(
	dialog_id: str,
	state: DialogSession,
	preference_pair: Optional[Tuple[int, Tuple[str, float], Tuple[str, float]]],
	system_role: str,
	output_path: Optional[Path] = None,
	persona_hint: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, str]]:
	"""Append the preference pair to preference_pair.json if it is new."""
	if not preference_pair:
		return None

	_, best_pair, worst_pair = preference_pair
	if not best_pair or not worst_pair:
		return None

	conversation = state.to_string_rep(keep_sys_da=False, keep_user_da=False)
	prompt_header = "You are the Persuader. Generate the Persuader reply that advances persuasion in a way that persuades the Persuadee to donate to Save the Children."
	persona_lines: List[str] = []
	if persona_hint:
		if persona_hint.get("personality"):
			persona_lines.append(f"Active cue: they show {persona_hint['personality']} traits.")
		if persona_hint.get("decision_making_style"):
			persona_lines.append(
				f"Decision tendency this turn: they favor a {persona_hint['decision_making_style']} style."
			)
	persona_block = ""
	if persona_lines:
		persona_block = "Persona hints about the Persuadee:\n" + "\n".join(persona_lines) + "\n"
	if conversation:
		prompt = f"{prompt_header}\n{persona_block}Conversation so far:\n{conversation}\n{system_role}:"
	else:
		prompt = f"{prompt_header}\n{persona_block}{system_role}:"

	# dialog_id = hashlib.sha1(hashable_state.encode("utf-8")).hexdigest()
	preference_entry = {
		"dialog_id": dialog_id,
		"prompt": prompt,
		"chosen": best_pair[0],
		"rejected": worst_pair[0],
	}
	if persona_hint:
		preference_entry["persona_hint"] = persona_hint
	anchor_dialog_id = getattr(state, "_anchor_dialog_id", None)
	if anchor_dialog_id:
		preference_entry["anchor_dialog_id"] = anchor_dialog_id

	output_path = output_path or Path(__file__).resolve().parents[1] / "preference_pair.jsonl"
	existing_entries = []
	if output_path.exists():
		try:
			with output_path.open("r", encoding="utf-8") as pref_file:
				raw_content = pref_file.read().strip()
			if raw_content:
				if raw_content.startswith("["):
					loaded = json.loads(raw_content)
					if isinstance(loaded, list):
						existing_entries = loaded
				else:
					for line in raw_content.splitlines():
						line = line.strip()
						if not line:
							continue
						try:
							entry = json.loads(line)
							if isinstance(entry, dict):
								existing_entries.append(entry)
						except json.JSONDecodeError:
							logger.warning("Failed to parse preference entry line; skipping.")
		except json.JSONDecodeError:
			logger.warning("%s is not valid JSON; resetting file.", output_path.name)
		except Exception as exc:
			logger.warning("Failed to read %s: %s", output_path.name, exc)

	def _entry_key(entry: Dict[str, str]) -> tuple[str, str, str, str]:
		return (
			entry.get("dialog_id", ""),
			entry.get("prompt", ""),
			entry.get("chosen", ""),
			entry.get("rejected", ""),
		)

	existing_keys = {_entry_key(entry) for entry in existing_entries}
	if _entry_key(preference_entry) not in existing_keys:
		existing_entries.append(preference_entry)
		with output_path.open("w", encoding="utf-8") as pref_file:
			for entry in existing_entries:
				pref_file.write(json.dumps(entry, ensure_ascii=True))
				pref_file.write("\n")
	return preference_entry
