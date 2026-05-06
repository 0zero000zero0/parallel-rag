import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, TypedDict

from src.clients import (
    BatchSearchDocs,
    OpenAIClient,
    RetrieverClient,
    RetrieverDocument,
)
from src.prompted_generation_base import (
    PromptedGenerationBase,
    build_openai_client_from_args,
    build_retriever_client_from_args,
    parse_stop_tokens,
    resolve_chat_template_components,
)

THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
SEARCH_QUERY_TAG_PATTERN = re.compile(
    r"<search>(.*?)</search>", flags=re.DOTALL | re.IGNORECASE
)
ANSWER_TAG_PATTERN = re.compile(
    r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE
)

IRCOT_SYSTEM_PROMPT = (
    "You are a reasoning assistant that can search for information to help answer questions accurately.\n"
    "You should think step by step. At each step, if you need external information, output a search query.\n"
    "Format your reasoning as follows:\n"
    "- First, think about the question inside <think>...</think>\n"
    "- If you need to search, output the search query inside <search>...</search>\n"
    "- The system will provide search results, then continue your reasoning\n"
    "- When you have enough information, provide the final answer inside <answer>...</answer>\n"
    "- In the final answer, the exact answer must be enclosed within <boxed></boxed> tag\n"
    "For example:\n"
    "<think>I need to find information about X</think>\n"
    "<search>query for X</search>\n"
    "[System provides search results]\n"
    "<think>Based on the search results, I can now answer...</think>\n"
    "<answer>The answer is <boxed>exact answer</boxed></answer>\n"
    "\n"
    "Important: Only search when you genuinely need external knowledge. If you already know the answer, provide it directly."
)


class IRCOTStep(TypedDict):
    iteration: int
    think: str
    search_query: str
    retrieved_docs: List[RetrieverDocument]


class IRCOTResult(TypedDict):
    query: str
    max_iterations: int
    executed_iterations: int
    search_count: int
    prompts: List[str]
    raw_outputs: List[str]
    steps: List[IRCOTStep]
    final_answer: str
    boxed_answer: str
    timing: Dict[str, Any]


class IRCOT(PromptedGenerationBase):
    def __init__(
        self,
        retriever: RetrieverClient,
        llm_client: OpenAIClient,
        max_search_limit: int = 5,
        max_iterations: int = 10,
        generation_max_tokens: int = 1024,
        generation_temperature: float = 0.7,
        generation_top_p: float = 0.9,
        stop_tokens: List[str] | None = None,
        use_chat_template: bool = False,
        tokenizer: Any = None,
    ):
        super().__init__(
            llm_client=llm_client,
            generation_max_tokens=generation_max_tokens,
            generation_temperature=generation_temperature,
            generation_top_p=generation_top_p,
            stop_tokens=stop_tokens,
            use_chat_template=use_chat_template,
            tokenizer=tokenizer,
        )
        self.retriever = retriever
        self.max_search_limit = max(1, max_search_limit)
        self.max_iterations = max(1, max_iterations)
        self.latest_batch_timing: Dict[str, Any] = {}

    @classmethod
    def from_args(cls, args) -> "IRCOT":
        use_chat_template, tokenizer = resolve_chat_template_components(args)
        retriever_client = build_retriever_client_from_args(args)
        llm_client = build_openai_client_from_args(
            args, use_chat_template=use_chat_template
        )

        max_search_limit = int(getattr(args, "max_search_limit", 5))
        max_iterations = int(getattr(args, "max_iterations", 10))
        generation_max_tokens = int(getattr(args, "generation_max_tokens", 1024))
        generation_temperature = float(getattr(args, "generation_temperature", 0.7))
        generation_top_p = float(getattr(args, "generation_top_p", 0.9))

        return cls(
            retriever=retriever_client,
            llm_client=llm_client,
            max_search_limit=max_search_limit,
            max_iterations=max_iterations,
            generation_max_tokens=generation_max_tokens,
            generation_temperature=generation_temperature,
            generation_top_p=generation_top_p,
            stop_tokens=parse_stop_tokens(args),
            use_chat_template=use_chat_template,
            tokenizer=tokenizer,
        )

    def _make_config(
        self, max_tokens: int, temperature: float, top_p: float
    ) -> SimpleNamespace:
        return SimpleNamespace(
            max_completion_length=max_tokens,
            temperature=temperature,
            top_p=top_p,
            vllm_n=1,
        )

    def _extract_search_query(self, text: str) -> str:
        matches = SEARCH_QUERY_TAG_PATTERN.findall(text)
        if not matches:
            return ""
        return matches[-1].strip()

    def _parse_answer(self, text: str) -> str:
        answer = self._extract_last_tag(ANSWER_TAG_PATTERN, text)
        return answer if answer else ""

    def _build_initial_prompt(self, question: str) -> Any:
        user_prompt = "\n\n".join(
            [
                self._format_external_context("Question", question),
                "Think step by step. If you need to search for information, use <search>query</search> tags.",
            ]
        )
        return self.format_prompt(IRCOT_SYSTEM_PROMPT, user_prompt)

    def _format_docs_block(self, docs: List[RetrieverDocument]) -> str:
        if not docs:
            return "No retrieved documents."
        return "\n\n".join(
            [
                f"DocID: {doc.get('id', '')}\nContent: {doc.get('contents', '')}"
                for doc in docs
            ]
        )

    def _generate_text_batch_custom(
        self, prompts: List[Any], config: SimpleNamespace
    ) -> List[str]:
        if not prompts:
            return []
        if not self.use_chat_template:
            prompt_texts = [self._prompt_to_text(prompt) for prompt in prompts]
            return self.llm_client.generate_text(prompt_texts, config, self.stop_tokens)
        return self.llm_client.generate_text(
            prompts, config, self.stop_tokens, tokenizer=self.tokenizer
        )

    @staticmethod
    def _now_ns() -> int:
        return time.perf_counter_ns()

    @staticmethod
    def _ns_to_ms(duration_ns: int) -> float:
        return duration_ns / 1_000_000.0

    def run(self, question: str) -> IRCOTResult:
        return self.run_batch([question])[0]

    def run_batch(self, questions: List[str]) -> List[IRCOTResult]:
        if not questions:
            return []

        run_batch_started_ns = self._now_ns()
        batch_phase_totals: Dict[str, float] = {
            "phase1_reasoning_ms": 0.0,
            "phase2_retrieval_ms": 0.0,
        }

        results: List[IRCOTResult] = [
            {
                "query": q,
                "max_iterations": self.max_iterations,
                "executed_iterations": 0,
                "search_count": 0,
                "prompts": [],
                "raw_outputs": [],
                "steps": [],
                "final_answer": "",
                "boxed_answer": "",
                "timing": {
                    "phase1_reasoning_ms": 0.0,
                    "phase2_retrieval_ms": 0.0,
                    "total_ms": 0.0,
                    "executed_iterations": 0,
                },
            }
            for q in questions
        ]

        prompts: List[Any] = [self._build_initial_prompt(q) for q in questions]
        completed = [False] * len(questions)
        search_counts = [0] * len(questions)

        config = self._make_config(
            max_tokens=self.generation_max_tokens,
            temperature=self.generation_temperature,
            top_p=self.generation_top_p,
        )

        for iteration in range(1, self.max_iterations + 1):
            active_indices = [i for i, c in enumerate(completed) if not c]
            if not active_indices:
                break

            active_prompts = [prompts[i] for i in active_indices]

            phase1_started_ns = self._now_ns()
            outputs = self._generate_text_batch_custom(active_prompts, config)
            phase1_ms = self._ns_to_ms(self._now_ns() - phase1_started_ns)
            batch_phase_totals["phase1_reasoning_ms"] += phase1_ms
            phase1_share = phase1_ms / len(active_indices)
            for sample_i in active_indices:
                results[sample_i]["timing"]["phase1_reasoning_ms"] += phase1_share

            queries_to_search: List[str] = []
            query_meta: List[tuple[int, str]] = []
            sample_output_map: Dict[int, str] = {}

            for local_i, sample_i in enumerate(active_indices):
                output = outputs[local_i]
                results[sample_i]["prompts"].append(self._prompt_to_text(prompts[sample_i]))
                results[sample_i]["raw_outputs"].append(output)
                results[sample_i]["executed_iterations"] += 1

                answer = self._parse_answer(output)
                if answer:
                    results[sample_i]["final_answer"] = answer
                    boxed = self._extract_boxed(answer)
                    results[sample_i]["boxed_answer"] = boxed if boxed else answer
                    completed[sample_i] = True
                    continue

                search_query = self._extract_search_query(output)
                if not search_query:
                    completed[sample_i] = True
                    if not results[sample_i]["final_answer"]:
                        results[sample_i]["final_answer"] = output.strip()
                    continue

                if search_counts[sample_i] >= self.max_search_limit:
                    completed[sample_i] = True
                    if not results[sample_i]["final_answer"]:
                        results[sample_i]["final_answer"] = output.strip()
                    continue

                search_counts[sample_i] += 1
                queries_to_search.append(search_query)
                query_meta.append((sample_i, search_query))
                sample_output_map[sample_i] = output

                if self.use_chat_template:
                    prompts[sample_i] = [
                        *prompts[sample_i],
                        {"role": "assistant", "content": output},
                    ]
                else:
                    prompts[sample_i] += output + "\n"

            if not queries_to_search:
                continue

            phase2_started_ns = self._now_ns()
            batch_docs = self.retriever.batch_search(queries_to_search)
            phase2_ms = self._ns_to_ms(self._now_ns() - phase2_started_ns)
            batch_phase_totals["phase2_retrieval_ms"] += phase2_ms
            phase2_samples = sorted({sample_i for sample_i, _ in query_meta})
            if phase2_samples:
                phase2_share = phase2_ms / len(phase2_samples)
                for sample_i in phase2_samples:
                    results[sample_i]["timing"]["phase2_retrieval_ms"] += phase2_share

            for idx, docs in enumerate(batch_docs):
                sample_i, search_query = query_meta[idx]
                docs_block = self._format_docs_block(docs)
                search_result_block = f"\n[Search Results for '{search_query}']:\n{docs_block}\n"

                results[sample_i]["steps"].append(
                    {
                        "iteration": iteration,
                        "think": self._extract_think(sample_output_map.get(sample_i, "")),
                        "search_query": search_query,
                        "retrieved_docs": docs,
                    }
                )

                if self.use_chat_template:
                    prompts[sample_i] = [
                        *prompts[sample_i],
                        {"role": "user", "content": search_result_block},
                    ]
                else:
                    prompts[sample_i] += search_result_block

        for idx, result in enumerate(results):
            result["search_count"] = search_counts[idx]
            if result["final_answer"]:
                continue
            if result["raw_outputs"]:
                last_output = result["raw_outputs"][-1]
                answer = self._parse_answer(last_output)
                if answer:
                    result["final_answer"] = answer
                    boxed = self._extract_boxed(answer)
                    result["boxed_answer"] = boxed if boxed else answer
                else:
                    result["final_answer"] = last_output.strip()
                    result["boxed_answer"] = self._extract_boxed(last_output)

        for result in results:
            result["timing"]["total_ms"] = (
                result["timing"]["phase1_reasoning_ms"]
                + result["timing"]["phase2_retrieval_ms"]
            )
            result["timing"]["executed_iterations"] = result["executed_iterations"]

        self.latest_batch_timing = {
            "num_questions": len(questions),
            "max_iterations": self.max_iterations,
            "total_ms": self._ns_to_ms(self._now_ns() - run_batch_started_ns),
            "phase_totals_ms": batch_phase_totals,
        }

        return results
