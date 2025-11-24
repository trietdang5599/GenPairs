import json
import logging
import re
from pathlib import Path

from collections import Counter
from typing import Dict, List, Tuple, Optional

from core.helpers import DialogSession
from core.gen_models import GenerationModel, DialogModel
from core.game import PersuasionGame
from utils.utils import log_prompt, format_messages_for_log
from utils.prompts import PERSUADEE_PROMPT_TEMPLATE
from utils.dialog_acts import USER_DIALOG_ACT_DEFINITIONS


logger = logging.getLogger(__name__)

class PersuadeeModel(DialogModel):
	def __init__(self,
			dialog_acts: List[str],
			inference_args: dict,
			backbone_model:GenerationModel, 
			conv_examples: List[DialogSession] = [], 
			max_hist_num_turns=5):
		super().__init__()
		self.conv_examples = conv_examples
		self.backbone_model = backbone_model
		self.dialog_acts = dialog_acts
		self.max_hist_num_turns = max_hist_num_turns
		self.persona_profiles = self._load_persona_profiles()
		self.da_definitions = {da: desc for da, desc in USER_DIALOG_ACT_DEFINITIONS.items() if da in dialog_acts}
		# prompts
		dialog_act_list = " ".join([f"[{da}]" for da in self.dialog_acts])
		self.task_prompt = PERSUADEE_PROMPT_TEMPLATE.format(
			dialog_act_list=dialog_act_list,
			examples=self.process_exp(),
		).replace("\t", "").strip()
		self.inference_args = inference_args
		self.classifier_args = {
			"max_new_tokens": 16,
			"temperature": 0.0,
			"do_sample": False,
			"return_full_text": False,
		}
		return

	def _load_persona_profiles(self) -> List[Dict[str, str]]:
		persona_path = Path(__file__).resolve().parents[1] / "data" / "bigfive_personas.jsonl"
		profiles: List[Dict[str, str]] = []
		if not persona_path.exists():
			logger.warning("Persona profile file not found at %s; continuing without personas.", persona_path)
			return profiles
		try:
			with persona_path.open("r", encoding="utf-8") as handle:
				for raw_line in handle:
					line = raw_line.strip()
					if not line:
						continue
					try:
						entry = json.loads(line)
					except json.JSONDecodeError:
						logger.warning("Failed to parse persona line: %s", line[:80])
						continue
					description = entry.get("description") or ""
					description = description.strip()
					if not description:
						continue
					profiles.append(
						{
							"description": description,
							"big_five": entry.get("big_five_personality", ""),
							"decision_making_style": entry.get("decision_making_style", ""),
						}
					)
		except Exception as exc:  # pragma: no cover
			logger.warning("Unable to load persona profiles from %s: %s", persona_path, exc)
		if not profiles:
			logger.warning("No persona descriptions loaded from %s.", persona_path)
		return profiles

	def _get_persona_profile(self, state: DialogSession) -> Optional[Dict[str, str]]:
		# Persona is assigned externally (e.g., during simulation); do not build/select here.
		return getattr(state, "_persona_profile", None)

	def _build_persona_context(self, persona_profile: Optional[Dict[str, str]]) -> str:
		if not persona_profile:
			return ""
		context_lines = ["Persona background for this conversation:", persona_profile["description"]]
		if persona_profile.get("big_five"):
			context_lines.append(f"Big-Five Personality: {persona_profile['big_five']}")
		if persona_profile.get("decision_making_style"):
			context_lines.append(f"Decision-Making Style: {persona_profile['decision_making_style']}")
		return "\n".join(context_lines) + "\n"

	def process_exp(self):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += exp.to_string_rep(keep_user_da=True) + "\n"
		return prompt_exps.strip()

	def _normalize_da(self, candidate: str) -> str | None:
		candidate = candidate.strip().lower()
		for da in self.dialog_acts:
			if candidate == da.lower():
				return da
		return None

	def _build_classification_segments(self, state: DialogSession, response: str) -> List[str]:
		response_text = response.strip()
		acts = ", ".join([f"[{da}]" for da in self.dialog_acts])
		definitions = "\n".join([f"[{da}] {desc}" for da, desc in self.da_definitions.items()])
		segments = []
		segments.append(f"Persuadee last response: {response_text}")
		segments.append(f"Dialog act definitions:\n{definitions}")
		segments.append(
			f"Select the single best dialog act label from {acts}. "
			"Answer with just the label in brackets (e.g., [donate], [neutral], [no donation])."
		)
		return segments

	def _classification_prompt(self, state: DialogSession, response: str) -> str:
		return "\n\n".join(self._build_classification_segments(state, response))

	def _classify_dialog_act(self, state: DialogSession, response: str) -> str | None:
		prompt = self._classification_prompt(state, response)
		try:
			data = self.backbone_model.generate(prompt, **self.classifier_args)
			label = self.backbone_model._cleaned_resp(data, prompt)[0].strip()
			normalized = self._normalize_da(label)
			if not normalized:
				normalized = self._normalize_da(label.replace("[", "").replace("]", ""))
			return normalized
		except Exception as exc:  # pragma: no cover - best effort fallback
			logger.debug("Failed to classify dialog act: %s", exc)
		return None
	
	def get_utterance(self, state:DialogSession, action=None) -> str:
		assert(state[-1][0] == PersuasionGame.SYS)
		action_instruction = ""
		if action and action in self.dialog_acts:
			action_instruction = (
				f"The selected dialog act for this turn is [{action}]. Respond using this dialog act, follow the format `[dialog_act] utterance`, and ensure the utterance contains at least one polite sentence.\n"
			)
		prompt = f"""
		{self.task_prompt}
		{action_instruction}
		{state.to_string_rep(keep_user_da=True, max_turn_to_display=self.max_hist_num_turns)}
		{self._build_persona_context(self._get_persona_profile(state))}
		Persuadee:
		"""
		prompt = prompt.replace("\t", "").strip()
		log_prompt(f"[PERSUADEE]\n{prompt}")
		# produce a response
		data = self.backbone_model.generate(prompt, **self.inference_args)
		user_resp = self.backbone_model._cleaned_resp(data, prompt)[0]
		return user_resp

	def get_utterance_w_da(self, state:DialogSession, action=None, classify: bool = False) -> "Tuple[str, str]":
		selected_da = action if action in self.dialog_acts else None
		user_resp = self.get_utterance(state, selected_da)
		start_idx = user_resp.find("[")
		end_idx = user_resp.find("]")
		parsed_da = None
		if start_idx != -1 and end_idx != -1 and end_idx > start_idx + 1:
			extracted = user_resp[start_idx + 1 : end_idx]
			user_resp = user_resp.replace(f"[{extracted}]", "", 1).strip()
			parsed_da = self._normalize_da(extracted)
		da = parsed_da or selected_da or PersuasionGame.U_Neutral
		need_classification = (
			classify
			or selected_da is not None
			or parsed_da is None
		)
		if need_classification:
			classified = self._classify_dialog_act(state, user_resp)
			if classified:
				da = classified
			elif parsed_da is None and selected_da:
				da = selected_da
		if self._looks_like_question(user_resp):
			da = PersuasionGame.U_Neutral
		return da, user_resp

	def _looks_like_question(self, text: str) -> bool:
		normalized = (text or "").strip().lower()
		if not normalized:
			return False
		if "?" in normalized:
			return True
		question_starters = (
			"can ",
			"could ",
			"would ",
			"will ",
			"what ",
			"why ",
			"how ",
			"when ",
			"where ",
			"who ",
			"is ",
			"are ",
			"do ",
			"does ",
		)
		return normalized.startswith(question_starters)

class PersuadeeChatModel(PersuadeeModel):
	def __init__(self,
			dialog_acts: List[str],
			inference_args: dict,
			backbone_model:GenerationModel, 
			conv_examples: List[DialogSession] = [], 
			max_hist_num_turns=5):
		super().__init__(
			dialog_acts=dialog_acts,
			inference_args=inference_args,
			backbone_model=backbone_model,
			conv_examples=conv_examples,
			max_hist_num_turns=max_hist_num_turns
		)
		self.inference_args = inference_args
		self.new_task_prompt = "The following is a new conversation between another Persuader and Persuadee."
		self.heuristic_args: dict = {
			"max_hist_num_turns": 2,
			"example_pred_turn": [[0, 2, 3, 4]]
		}
		self.classifier_args = {
			"max_new_tokens": 16,
			"temperature": 0.0,
			"do_sample": False,
			"return_full_text": False,
		}
		return
	
	def __proccess_chat_exp(self, exp:DialogSession, max_hist_num_turns: int = -1):
		if len(exp) == 0:
			return []
		# P4G dataset starts with the system
		assert(exp[0][0] == PersuasionGame.SYS)

		prompt_messages = []
		num_turns_to_truncate = 0
		if max_hist_num_turns > 0:
			num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)
		
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			if role == PersuasionGame.SYS:
				prompt_messages.append({
					"role": "user",
					"content": f"{role}: {utt}".strip()
				})
			else:
				prompt_messages.append({
					"role": "assistant",  # assistant is the user simulator
					"content": f"{role}: [{da}] {utt}".strip()
				})
		return prompt_messages
	
	def get_utterance(self, state:DialogSession, action=None) -> str:
		assert(state[-1][0] == PersuasionGame.SYS)  # next turn is user's turn
		messages = [
			{'role': 'system', 'content': self.task_prompt},
		]
		if action and action in self.dialog_acts:
			messages.append({
				'role': 'system',
				'content': (
					f"The selected dialog act for this turn is [{action}]. Respond using this dialog act and follow the format `[dialog_act] utterance`."
				)
			})
		messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)
		persona_context = self._build_persona_context(self._get_persona_profile(state))
		if persona_context:
			messages.append({'role': 'system', 'content': persona_context.strip()})
		log_prompt(f"[PERSUADEE_CHAT]\n{format_messages_for_log(messages)}")

		# produce a response
		data = self.backbone_model.chat_generate(messages, **self.inference_args)
		user_resp = self.backbone_model._cleaned_chat_resp(
			data, assistant_role=f"{PersuasionGame.USR}:", user_role=f"{PersuasionGame.SYS}:"
		)[0]
		return user_resp
	
	def get_utterance_from_batched_states(self, states:List[DialogSession], action=None) -> List[str]:
		assert(all([state[-1][0] == PersuasionGame.SYS for state in states]))
		all_prompts: List[List[dict]] = []
		for idx, state in enumerate(states):
			messages = [
				{'role': 'system', 'content': self.task_prompt},
			]
			if action and action in self.dialog_acts:
				messages.append({
					'role': 'system',
					'content': (
						f"The selected dialog act for this turn is [{action}]. Respond using this dialog act and follow the format `[dialog_act] utterance`."
					)
				})
			messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)
			log_prompt(f"[PERSUADEE_CHAT_BATCH_{idx}]\n{format_messages_for_log(messages)}")
			all_prompts.append(messages)
		# add persona context after conversation history to avoid breaking ordering
		for idx, messages in enumerate(all_prompts):
			persona_context = self._build_persona_context(self._get_persona_profile(states[idx]))
			if persona_context:
				messages.append({'role': 'system', 'content': persona_context.strip()})
		# produce a response
		datas = self.backbone_model.chat_generate_batched(all_prompts, **self.inference_args)
		user_resps = []
		for data in datas:
			user_resp = self.backbone_model._cleaned_chat_resp(
				data, assistant_role=f"{PersuasionGame.USR}:", user_role=f"{PersuasionGame.SYS}:"
			)
			user_resps.append(user_resp[0])
		return user_resps

	def _classify_dialog_act(self, state: DialogSession, response: str) -> str | None:
		content = "\n\n".join(self._build_classification_segments(state, response))
		messages = [
			{
				"role": "system",
				"content": (
					"You are an annotator who labels persuadee dialog acts. "
					"Use the provided definitions to choose the single best label."
				),
			},
			{
				"role": "user",
				"content": content,
			},
		]
		try:
			data = self.backbone_model.chat_generate(messages, **self.classifier_args)
			label = self.backbone_model._cleaned_chat_resp(data, assistant_role="", user_role="")[0].strip()
			normalized = self._normalize_da(label) or self._normalize_da(label.replace("[", "").replace("]", ""))
			return normalized
		except Exception as exc:  # pragma: no cover - best effort fallback
			logger.debug("Failed to classify dialog act (chat): %s", exc)
		return None
	
	def get_utterance_w_da_from_batched_states(self, states:List[DialogSession], action=None):
		gen_user_resps = self.get_utterance_from_batched_states(states, action)
		das = []
		user_resps = []
		# extract da
		for user_resp in gen_user_resps:
			start_idx = user_resp.find("[")
			end_idx = user_resp.find("]")
			if start_idx == -1 or end_idx == -1:
				da = PersuasionGame.U_Neutral
			else:
				da = user_resp[start_idx+1:end_idx]
				user_resp = user_resp.replace(f"[{da}]", "", 1).strip()
				if da not in self.dialog_acts:
					da = PersuasionGame.U_Neutral
			das.append(da)
			user_resps.append(user_resp)
		return das, user_resps

	def __process_heuristics_chat_exp(self, dialog:DialogSession):
		if len(dialog) == 0:
			return []
		# assumes you start with the system
		# and ends with a user utterance to predict
		assert(dialog[0][0] == PersuasionGame.SYS)
		assert(dialog[-1][0] == PersuasionGame.USR)

		prompt_messages = []
		input_context = []
		answer_da = dialog[-1][1]
		for i, (role, da, utt) in enumerate(dialog):
			# if assistant is the Persuader, then current data is also Persuader -> then it is of role "system"
			# treat this as a task
			content = f"{role}: {utt}".strip()
			input_context.append(content)
		input_context.append(f"{dialog.USR} feeling:")

		prompt_q = "\n".join(input_context)
		prompt_messages.append({
			"role": 'user',
			"content": prompt_q
		})
		prompt_messages.append({
			"role": 'assistant',
			"content": f"{answer_da}"
		})
		return prompt_messages
	
	def __truncate_heuristics_dialog(self, dialog:DialogSession, pred_end_idx=-1):
		max_history_length = self.heuristic_args['max_hist_num_turns']
		if pred_end_idx == -1:
			pred_end_idx = len(dialog.history) - 1
		new_sys_start_idx = max(0, pred_end_idx - (max_history_length * 2 - 1))
		new_history = []
		for j, (role, da, utt) in enumerate(dialog):
			if j >= new_sys_start_idx:
				new_history.append((role, da, utt))
			if j == pred_end_idx:
				# user's utternace to predict
				break
		new_dialog_session = DialogSession(dialog.SYS, dialog.USR).from_history(new_history)
		return new_dialog_session
	
	def process_heurstics_chat_exp(self, new_task_prompt: str):
		prompt_exps = []
		for i, exp in enumerate(self.conv_examples):
			pred_end_turns: List[int] = self.heuristic_args['example_pred_turn'][i]
			# make a new dialogue session until that pred_idx with max max_history_length turns
			for pred_end_turn in pred_end_turns:
				pred_end_idx = pred_end_turn * 2 + 1
				new_dialog_session = self.__truncate_heuristics_dialog(exp, pred_end_idx)
				prompt_exps += self.__process_heuristics_chat_exp(new_dialog_session)
				prompt_exps.append({
					"role":"system", "content": new_task_prompt
				})
		return prompt_exps[:-1]

	def predict_da(self, state:DialogSession, never_end=True) -> str:
		# never_end=True  during real chat, let user choose to terminate, not this function
		# insert prop to donate, and compute the likelihood of user simulator agreeing to donate
		assert(state[-1][0] == PersuasionGame.USR)

		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.process_heurstics_chat_exp(new_task_prompt=self.new_task_prompt),
			{'role': 'system', 'content': self.new_task_prompt}
		]
		new_dialog_session = self.__truncate_heuristics_dialog(state, -1)
		messages += self.__process_heuristics_chat_exp(new_dialog_session)[:-1]

		# majority vote, same as value function
		inf_args = {
			"max_new_tokens": 5,
			"temperature": 0.7,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 5,
		}
		datas = self.backbone_model.chat_generate(messages, **inf_args)
		# process into das
		sampled_das: list = []
		for resp in datas:
			user_da = resp['generated_text'].strip()
			if user_da not in self.dialog_acts:
				sampled_das.append(PersuasionGame.U_Neutral)
			if never_end:
				if user_da == PersuasionGame.U_Donate:
					sampled_das.append(PersuasionGame.U_PositiveReaction)
				elif user_da == PersuasionGame.U_NoDonation:
					sampled_das.append(PersuasionGame.U_NegativeReaction)
				else:
					sampled_das.append(user_da)
			else:
				sampled_das.append(user_da)
		logger.info(f"sampled das: {sampled_das}")
		# majority vote
		counted_das = Counter(sampled_das)
		user_da = counted_das.most_common(1)[0][0]
		return user_da

__all__ = [
	"PersuadeeModel",
	"PersuadeeChatModel",
]
