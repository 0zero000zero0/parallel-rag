import json
from types import SimpleNamespace
from typing import Any, Dict, List

from transformers import AutoTokenizer

from src.clients import OpenAIClient


def parse_headers(header_args: List[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for item in header_args:
        if "=" not in item:
            raise ValueError(
                f"Invalid header format: {item!r}. Use KEY=VALUE")
        key, value = item.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers


def load_chat_messages(params: Dict[str, Any]) -> List[Dict[str, str]]:
    messages_json = params.get("messages_json")
    if messages_json:
        with open(messages_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("messages_json must contain a list of messages")
        return data

    messages: List[Dict[str, str]] = []
    if params.get("system"):
        messages.append({"role": "system", "content": params["system"]})
    messages.append({"role": "user", "content": params["user"]})
    return messages


def to_chat_prompt(system_content: str, user_content: str) -> Any:
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


tokenizer = AutoTokenizer.from_pretrained(
    "/home/zdw2200170271/llm/models/Qwen3-32B")

PARAMS: Dict[str, Any] = {
    "base_url": "http://127.0.0.1:8000",
    "model": "Qwen3-32B",
    "api_key": "TEST",
    "endpoint": None,
    "timeout": 120.0,
    "headers": [],
    "max_completion_length": 512,
    "temperature": 0.7,
    "top_p": 0.8,
    "vllm_n": 1,
    "stop_tokens": ["</search>", "</answer>"],
    "include_stop_str_in_output": True,
    "prompt": "You are tasked to rewrite or improve the main search query into one better query according to original question. You must conduct reasoning inside <think> and </think> first and output your search query between <search> and </search>. Input paramters\n- Original Question: Who plays the doctor in dexter season 1?\n- Main Agent Search Query: Who plays the doctor in Dexter season 1. \n\n Only output the reasoning process first and search query succedent",
    # "enable_thinking": False
}

# chat ml 格式
# system_prompt = (
#     "You are a path search agent on parallel mode. "
#     "You are tasked to rewrite or improve the main search query into one better query. "
#     "You must conduct reasoning inside <think> and </think> first and output the new search query by <search> query here </search>. "
# )
# original_question = "Who plays the doctor in dexter season 1?"
# main_query = " Who plays the doctor in Dexter season 1."
# user_prompt = (
#     f"You must use the following two inputs to produce your next query:\n"
#     f"- Original Question: {original_question}\n"
#     f"- Main Agent Search Query: {main_query}\n\n"
#     # f"Main Agent Output:\n{state_prompt}"
# )
# chat_ml = to_chat_prompt(system_prompt, user_prompt)
# chat_ml_str = tokenizer.apply_chat_template(chat_ml, tokenize=False)
# PARAMS['prompt'] = chat_ml_str


def main() -> None:
    params = PARAMS

    extra_headers = parse_headers(
        params["headers"]) if params["headers"] else None
    client = OpenAIClient(base_url=params["base_url"],
                          model=params["model"],
                          api_key=params["api_key"],
                          timeout=params["timeout"],
                          endpoint=params["endpoint"],
                          extra_headers=extra_headers)

    config = SimpleNamespace(max_completion_length=params["max_completion_length"],
                             temperature=params["temperature"],
                             top_p=params["top_p"],
                             vllm_n=params["vllm_n"])

    prompts = [params["prompt"]]

    outputs = client.generate(prompts=prompts,
                              config=config,
                              stop_tokens=params["stop_tokens"])

    # print("=== Request Summary ===")
    # print(f"base_url: {params['base_url']}")
    # print(f"model: {params['model']}")
    # print(f"is_chat: {params['is_chat']}")
    # print(f"endpoint: {params['endpoint']}")
    # print(f"timeout: {params['timeout']}")
    # print(f"max_completion_length: {params['max_completion_length']}")
    # print(f"temperature: {params['temperature']}")
    # print(f"top_p: {params['top_p']}")
    # print(f"vllm_n: {params['vllm_n']}")
    # print(f"stop_tokens: {params['stop_tokens']}")
    # print()

    print("=== Response ===")
    if not outputs:
        print("No outputs returned (request may have failed).")
        return

    for i, item in enumerate(outputs, start=1):
        print(f"[{i}] finish_reason={item.get('finish_reason')}")
        print(item.get("text", ""))
        print("-" * 60)


if __name__ == "__main__":
    main()
