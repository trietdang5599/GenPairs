from core.P4GSystemPlanner import P4GChatSystemPlanner, P4GSystemPlanner
from core.PersuadeeModel import PersuadeeChatModel, PersuadeeModel
from core.PersuaderModel import PersuaderChatModel, PersuaderModel
from core.gen_models import (
	LocalModel,
	OpenAIChatModel,
	OpenAIModel,
)


def create_factor_llm(cmd_args):
	if cmd_args.llm in ["gpt-3.5-turbo", "gpt-3.5-turbo-1106"]:
		backbone_model = OpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm in ["qwen2.5-0.5b", "qwen2.5-7b", "llamda-3-8b", "deepseek-r1"]:
		if cmd_args.llm == "qwen2.5-7b":
			model_source = "Qwen/Qwen2.5-7B-Instruct"
		elif cmd_args.llm == "qwen2.5-0.5b":
			model_source = "Qwen/Qwen2.5-0.5B-Instruct"
		elif cmd_args.llm == "llamda-3-8b":
			model_source = "meta-llama/Meta-Llama-3-8B-Instruct"
		else:
			model_source = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
		backbone_model = LocalModel(model_source, trust_remote_code=True)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	# elif cmd_args.llm == "local":
	# 	if not cmd_args.local_model_path:
	# 		raise ValueError("--local-model-path is required when --llm local")
	# 	model_kwargs = {}
	# 	if getattr(cmd_args, "local_base_model", ""):
	# 		model_kwargs["base_model_name_or_path"] = cmd_args.local_base_model
	# 	backbone_model = LocalModel(
	# 		cmd_args.local_model_path,
	# 		trust_remote_code=cmd_args.local_trust_remote_code,
	# 		model_kwargs=model_kwargs or None,
	# 	)
	# 	SysModel = PersuaderChatModel
	# 	UsrModel = PersuadeeChatModel
	# 	SysPlanner = P4GChatSystemPlanner
	else:
		raise ValueError(f"Unsupported llm: {cmd_args.llm}")

	return backbone_model, SysModel, UsrModel, SysPlanner
