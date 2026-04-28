from typing import Any, List, TypedDict

from src.prompted_generation_base import (PromptedGenerationBase,
                                          build_openai_client_from_args,
                                          parse_stop_tokens,
                                          resolve_chat_template_components)

COT_SYSTEM_PROMPT = (
    "You are a helpful assistant that can solve the given question step by step. "
    "Given a question, you need to first think about the reasoning process in the mind and then provide the answer. "
    "The reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tag respectively. "
    "For example, <think> This is the reasoning process. </think> <answer> The final answer is  <boxed> showt answer only </boxed> </answer>. "
    "In the last part of the answer, the final exact answer must be enclosed within <boxed></boxed> tag."
)


class CotResult(TypedDict):
    query: str
    prompt: str
    raw_output: str
    think: str
    final_answer: str
    boxed_answer: str


class Cot(PromptedGenerationBase):
    @classmethod
    def from_args(cls, args) -> "Cot":
        use_chat_template, tokenizer = resolve_chat_template_components(args)
        llm_client = build_openai_client_from_args(
            args, use_chat_template=use_chat_template)
        return cls(llm_client=llm_client,
                   generation_max_tokens=int(
                       getattr(args, "generation_max_tokens", 1024)),
                   generation_temperature=float(
                       getattr(args, "generation_temperature", 0.7)),
                   generation_top_p=float(getattr(args,
                                                  "generation_top_p",
                                                  0.9)),
                   stop_tokens=parse_stop_tokens(args),
                   use_chat_template=use_chat_template,
                   tokenizer=tokenizer)

    def _build_prompt(self, question: str) -> Any:
        user_prompt = "\n\n".join([
            self._format_external_context("Question", question),
            "Think step by step and then answer the question.",
        ])
        return self.format_prompt(COT_SYSTEM_PROMPT, user_prompt)

    def run(self, question: str) -> CotResult:
        return self.run_batch([question])[0]

    def run_batch(self, questions: List[str]) -> List[CotResult]:
        prompts = [self._build_prompt(question) for question in questions]
        outputs = self._generate_text_batch(prompts)

        results: List[CotResult] = []
        for question, prompt, output in zip(questions, prompts, outputs):
            results.append({
                "query": question,
                "prompt": self._prompt_to_text(prompt),
                "raw_output": output,
                "think": self._extract_think(output),
                "final_answer": self._extract_answer(output),
                "boxed_answer": self._extract_boxed(output),
            })
        return results
