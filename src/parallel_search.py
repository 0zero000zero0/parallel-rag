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
from src.parallel_rag import _coerce_bool, _first_not_none, _load_tokenizer
from src.prompted_generation_base import (
    PromptedGenerationBase,
    build_retriever_client_from_args,
    parse_stop_tokens,
)

THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
SEARCH_TAG_PATTERN = re.compile(
    r"<search>(.*?)</search>", flags=re.DOTALL | re.IGNORECASE
)
ANSWER_TAG_PATTERN = re.compile(
    r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE
)


class ParallelSearchResult(TypedDict):
    query: str
    max_iterations: int
    executed_iterations: int
    navigator_agent_prompts: List[str]
    navigator_agent_thoughts: List[str]
    search_queries: List[List[str]]
    retrieved_docs: List[BatchSearchDocs]
    search_information: List[str]
    final_answer: str
    timing: Dict[str, Any]


class ParallelSearch(PromptedGenerationBase):
    def __init__(
        self,
        retriever: RetrieverClient,
        navigator_agent_llm_client: OpenAIClient,
        navigator_agent_max_tokens: int = 512,
        navigator_agent_temperature: float = 0.6,
        navigator_agent_top_p: float = 0.9,
        synthesize_max_tokens: int = 768,
        synthesize_temperature: float = 0.3,
        synthesize_top_p: float = 0.9,
        stop_tokens: List[str] | None = None,
        navigator_agent_use_chat_template: bool = False,
        navigator_agent_tokenizer: Any = None,
    ):
        super().__init__(
            llm_client=navigator_agent_llm_client,
            generation_max_tokens=synthesize_max_tokens,
            generation_temperature=synthesize_temperature,
            generation_top_p=synthesize_top_p,
            stop_tokens=stop_tokens,
            use_chat_template=navigator_agent_use_chat_template,
            tokenizer=navigator_agent_tokenizer,
        )
        self.retriever = retriever
        self.navigator_agent_llm_client = navigator_agent_llm_client
        self.navigator_agent_max_tokens = navigator_agent_max_tokens
        self.navigator_agent_temperature = navigator_agent_temperature
        self.navigator_agent_top_p = navigator_agent_top_p
        self.synthesize_max_tokens = synthesize_max_tokens
        self.synthesize_temperature = synthesize_temperature
        self.synthesize_top_p = synthesize_top_p
        self.navigator_agent_use_chat_template = navigator_agent_use_chat_template
        self.navigator_agent_tokenizer = navigator_agent_tokenizer

        self.latest_batch_timing: Dict[str, Any] = {}

    @classmethod
    def from_args(cls, args) -> "ParallelSearch":
        navigator_agent_use_chat_template = _coerce_bool(
            _first_not_none(
                getattr(args, "navigator_agent_use_chat_template", None),
                getattr(args, "shared_use_chat_template", None),
                getattr(args, "use_chat_template", None),
            ),
            default=False,
        )

        navigator_agent_model = _first_not_none(
            getattr(args, "navigator_agent_model", None),
            getattr(args, "shared_model", None),
            getattr(args, "model", None),
            getattr(args, "llm_model", None),
            "Qwen3-14B",
        )

        navigator_agent_model_path = _first_not_none(
            getattr(args, "navigator_agent_model_path", None),
            getattr(args, "shared_model_path", None),
            getattr(args, "model_path", None),
            navigator_agent_model,
        )

        navigator_agent_tokenizer = _first_not_none(
            getattr(args, "navigator_agent_tokenizer", None),
            getattr(args, "shared_tokenizer", None),
        )
        if navigator_agent_use_chat_template and navigator_agent_tokenizer is None:
            navigator_agent_tokenizer = _load_tokenizer(navigator_agent_model_path)

        retriever_client = build_retriever_client_from_args(args)
        navigator_agent_llm_client = OpenAIClient(
            base_url=_first_not_none(
                getattr(args, "navigator_agent_openai_base_url", None),
                getattr(args, "shared_openai_base_url", None),
                getattr(args, "openai_base_url", None),
                "http://127.0.0.1:8001",
            ),
            model=navigator_agent_model,
            api_key=_first_not_none(
                getattr(args, "navigator_agent_openai_api_key", None),
                getattr(args, "shared_openai_api_key", None),
                getattr(args, "openai_api_key", None),
                "TEST",
            ),
            timeout=_first_not_none(
                getattr(args, "navigator_agent_llm_timeout", None),
                getattr(args, "shared_llm_timeout", None),
                getattr(args, "llm_timeout", None),
            ),
            use_chat_template=navigator_agent_use_chat_template,
        )

        navigator_agent_max_tokens = int(
            getattr(args, "navigator_agent_max_tokens", 256)
        )
        navigator_agent_temperature = float(
            getattr(args, "navigator_agent_temperature", 0.6)
        )
        navigator_agent_top_p = float(getattr(args, "navigator_agent_top_p", 0.9))

        synthesize_max_tokens = int(getattr(args, "synthesize_max_tokens", 768))
        synthesize_temperature = float(getattr(args, "synthesize_temperature", 0.3))
        synthesize_top_p = float(getattr(args, "synthesize_top_p", 0.9))

        return cls(
            retriever=retriever_client,
            navigator_agent_llm_client=navigator_agent_llm_client,
            navigator_agent_max_tokens=navigator_agent_max_tokens,
            navigator_agent_temperature=navigator_agent_temperature,
            navigator_agent_top_p=navigator_agent_top_p,
            synthesize_max_tokens=synthesize_max_tokens,
            synthesize_temperature=synthesize_temperature,
            synthesize_top_p=synthesize_top_p,
            stop_tokens=parse_stop_tokens(args),
            navigator_agent_use_chat_template=navigator_agent_use_chat_template,
            navigator_agent_tokenizer=navigator_agent_tokenizer,
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

    def _format_prompt_by_template(
        self, system_prompt: str, user_prompt: str, use_chat_template: bool
    ) -> Any:
        if not use_chat_template:
            return f"{system_prompt}\n\n{user_prompt}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _split_search_query(self, raw_query: str) -> List[str]:
        parts = [part.strip() for part in raw_query.split("##")]
        return [part for part in parts if part]

    def _extract_searches(self, text: str) -> List[str]:
        searches: List[str] = []
        for matched in SEARCH_TAG_PATTERN.findall(text):
            for query in self._split_search_query(matched.strip()):
                if query and query not in searches:
                    searches.append(query)
        return searches

    def _parse_answer(self, text: str) -> str:
        return self._extract_last_tag(ANSWER_TAG_PATTERN, text)

    def _extract_think(self, text: str) -> str:
        extracted = self._extract_last_tag(THINK_TAG_PATTERN, text)
        return extracted if extracted else text.strip()

    def _build_navigator_agent_prompt(
        self,
        question: str,
        history_blocks: List[str],
        use_chat_template: bool,
    ) -> Any:
        information_history = "\n\n".join(history_blocks) if history_blocks else "None"

        system_prompt = (
            "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. "
            "If the original query is complex or involves multiple parts, you are encouraged to decompose it into smaller sub-questions, separated by ##. "
            "For example: <search> sub-question 1 ## sub-question 2 </search>. "
            "You can search as many times as you want. "
            "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. "
            "For example, <answer> xxx </answer>."
        )
        user_prompt = "\n\n".join(
            [
                self._format_external_context("Question", question),
                self._format_external_context(
                    "Retrieved Information History", information_history
                ),
                "Now continue reasoning. If needed, output <search>...</search>; otherwise output <answer>...</answer>.",
            ]
        )
        return self._format_prompt_by_template(
            system_prompt, user_prompt, use_chat_template
        )

    def _build_information_appendix(
        self, search_queries: List[str], docs_per_query: BatchSearchDocs
    ) -> str:
        if not search_queries:
            return "\n<information>\nNo query issued.\n</information>\n"

        blocks: List[str] = []
        for query_id, query in enumerate(search_queries, start=1):
            docs = (
                docs_per_query[query_id - 1]
                if query_id - 1 < len(docs_per_query)
                else []
            )
            if docs:
                doc_lines: List[str] = []
                for doc in docs:
                    doc_id = str(doc.get("id", "")).strip()
                    contents = str(doc.get("contents", "")).strip()
                    if doc_id:
                        doc_lines.append(f"[{doc_id}] {contents}")
                    else:
                        doc_lines.append(contents)
                doc_text = "\n".join(doc_lines)
            else:
                doc_text = "No relevant results."

            blocks.append(
                "\n".join(
                    [
                        f"Query {query_id}: {query}",
                        doc_text,
                    ]
                )
            )

        return "\n<information>\n" + "\n\n".join(blocks) + "\n</information>\n"

    def _build_final_answer_prompt(
        self, question: str, history_blocks: List[str], use_chat_template: bool
    ) -> Any:
        information_history = "\n\n".join(history_blocks) if history_blocks else "None"
        system_prompt = (
            "Answer the given question based on available information. "
            "Output only concise final answer inside <answer>...</answer>."
        )
        user_prompt = "\n\n".join(
            [
                self._format_external_context("Question", question),
                self._format_external_context(
                    "Retrieved Information History", information_history
                ),
            ]
        )
        return self._format_prompt_by_template(
            system_prompt, user_prompt, use_chat_template
        )

    def _generate_text_batch(
        self,
        prompts: List[Any],
        config: SimpleNamespace,
        llm_client: OpenAIClient,
        tokenizer: Any = None,
        use_chat_template: bool | None = None,
    ) -> List[str]:
        if not prompts:
            return []

        active_use_chat_template = (
            self.use_chat_template
            if use_chat_template is None
            else bool(use_chat_template)
        )

        if not active_use_chat_template:
            prompt_texts = [self._prompt_to_text(p) for p in prompts]
            return llm_client.generate_text(prompt_texts, config, self.stop_tokens)

        return llm_client.generate_text(
            prompts, config, self.stop_tokens, tokenizer=tokenizer
        )

    @staticmethod
    def _now_ns() -> int:
        return time.perf_counter_ns()

    @staticmethod
    def _ns_to_ms(duration_ns: int) -> float:
        return duration_ns / 1_000_000.0

    def _init_batch_runtime(
        self, questions: List[str], max_iterations: int
    ) -> tuple[List[bool], List[ParallelSearchResult], List[List[str]]]:
        completed = [False] * len(questions)

        results: List[ParallelSearchResult] = [
            {
                "query": q,
                "max_iterations": max_iterations,
                "executed_iterations": 0,
                "navigator_agent_prompts": [],
                "navigator_agent_thoughts": [],
                "search_queries": [],
                "retrieved_docs": [],
                "search_information": [],
                "final_answer": "",
                "timing": {
                    "phase1_navigator_ms": 0.0,
                    "phase2_retrieval_ms": 0.0,
                    "phase3_finalize_ms": 0.0,
                    "total_ms": 0.0,
                    "executed_iterations": 0,
                },
            }
            for q in questions
        ]

        information_histories: List[List[str]] = [[] for _ in questions]
        return completed, results, information_histories

    def run(self, question: str, max_iterations: int = 4) -> ParallelSearchResult:
        return self.run_batch([question], max_iterations=max_iterations)[0]

    def run_batch(
        self, questions: List[str], max_iterations: int = 5
    ) -> List[ParallelSearchResult]:
        if not questions:
            return []

        run_batch_started_ns = self._now_ns()
        batch_phase_totals: Dict[str, float] = {
            "phase1_navigator_ms": 0.0,
            "phase2_retrieval_ms": 0.0,
            "phase3_finalize_ms": 0.0,
        }
        iteration_timings: List[Dict[str, Any]] = []

        completed, results, information_histories = self._init_batch_runtime(
            questions=questions, max_iterations=max_iterations
        )

        navigator_agent_config = self._make_config(
            max_tokens=self.navigator_agent_max_tokens,
            temperature=self.navigator_agent_temperature,
            top_p=self.navigator_agent_top_p,
        )

        for iteration_idx in range(max_iterations):
            iteration_started_ns = self._now_ns()
            iteration_timing: Dict[str, Any] = {
                "iteration": iteration_idx + 1,
                "active_samples": 0,
                "search_query_count": 0,
                "phase1_navigator_ms": 0.0,
                "phase2_retrieval_ms": 0.0,
                "iteration_total_ms": 0.0,
            }

            active_indices = [i for i, c in enumerate(completed) if not c]
            if not active_indices:
                break

            iteration_timing["active_samples"] = len(active_indices)

            active_prompts = [
                self._build_navigator_agent_prompt(
                    question=questions[i],
                    history_blocks=information_histories[i],
                    use_chat_template=self.navigator_agent_use_chat_template,
                )
                for i in active_indices
            ]

            phase1_started_ns = self._now_ns()
            navigator_outputs = self._generate_text_batch(
                active_prompts,
                navigator_agent_config,
                llm_client=self.navigator_agent_llm_client,
                tokenizer=self.navigator_agent_tokenizer,
                use_chat_template=self.navigator_agent_use_chat_template,
            )
            phase1_ms = self._ns_to_ms(self._now_ns() - phase1_started_ns)
            batch_phase_totals["phase1_navigator_ms"] += phase1_ms
            iteration_timing["phase1_navigator_ms"] = phase1_ms
            phase1_share = phase1_ms / len(active_indices)

            for sample_i in active_indices:
                results[sample_i]["timing"]["phase1_navigator_ms"] += phase1_share

            sample_search_queries: List[List[str]] = [[] for _ in range(len(questions))]
            flat_queries: List[str] = []
            query_meta: List[tuple[int, int]] = []

            for local_i, sample_i in enumerate(active_indices):
                navigator_output = navigator_outputs[local_i]
                if not results[sample_i]["navigator_agent_prompts"]:
                    results[sample_i]["navigator_agent_prompts"].append(
                        self._prompt_to_text(active_prompts[local_i])
                    )
                results[sample_i]["navigator_agent_prompts"].append(navigator_output)
                results[sample_i]["executed_iterations"] += 1

                navigator_think = self._extract_think(navigator_output)
                results[sample_i]["navigator_agent_thoughts"].append(navigator_think)

                answer = self._parse_answer(navigator_output)
                if answer:
                    results[sample_i]["final_answer"] = answer
                    completed[sample_i] = True
                    continue

                search_queries = self._extract_searches(navigator_output)
                results[sample_i]["search_queries"].append(search_queries)
                sample_search_queries[sample_i] = search_queries
                if not search_queries:
                    completed[sample_i] = True
                    continue

                for query_id, query in enumerate(search_queries, start=1):
                    flat_queries.append(query)
                    query_meta.append((sample_i, query_id))

            iteration_timing["search_query_count"] = len(flat_queries)
            if not flat_queries:
                iteration_timing["iteration_total_ms"] = self._ns_to_ms(
                    self._now_ns() - iteration_started_ns
                )
                iteration_timings.append(iteration_timing)
                continue

            phase2_started_ns = self._now_ns()
            flat_docs = self.retriever.batch_search(flat_queries)
            phase2_ms = self._ns_to_ms(self._now_ns() - phase2_started_ns)
            batch_phase_totals["phase2_retrieval_ms"] += phase2_ms
            iteration_timing["phase2_retrieval_ms"] = phase2_ms

            retrieval_samples = sorted({sample_i for sample_i, _ in query_meta})
            if retrieval_samples:
                retrieval_share = phase2_ms / len(retrieval_samples)
                for sample_i in retrieval_samples:
                    results[sample_i]["timing"]["phase2_retrieval_ms"] += (
                        retrieval_share
                    )

            docs_map: Dict[tuple[int, int], List[RetrieverDocument]] = {}
            for idx, docs in enumerate(flat_docs):
                docs_map[query_meta[idx]] = docs

            for sample_i in active_indices:
                search_queries = sample_search_queries[sample_i]
                if not search_queries:
                    continue

                docs_per_query: BatchSearchDocs = [
                    docs_map.get((sample_i, query_id), [])
                    for query_id, _ in enumerate(search_queries, start=1)
                ]
                results[sample_i]["retrieved_docs"].append(docs_per_query)

                information_appendix = self._build_information_appendix(
                    search_queries=search_queries,
                    docs_per_query=docs_per_query,
                )
                results[sample_i]["search_information"].append(information_appendix)
                information_histories[sample_i].append(information_appendix)
                results[sample_i]["navigator_agent_prompts"].append(
                    information_appendix
                )

            iteration_timing["iteration_total_ms"] = self._ns_to_ms(
                self._now_ns() - iteration_started_ns
            )
            iteration_timings.append(iteration_timing)

        for idx, result in enumerate(results):
            if result["final_answer"]:
                continue

            final_prompt = self._build_final_answer_prompt(
                question=questions[idx],
                history_blocks=information_histories[idx],
                use_chat_template=self.navigator_agent_use_chat_template,
            )
            if not result["navigator_agent_prompts"]:
                result["navigator_agent_prompts"].append(
                    self._prompt_to_text(final_prompt)
                )

            phase3_started_ns = self._now_ns()
            final_outputs = self._generate_text_batch(
                [final_prompt],
                self._make_config(
                    max_tokens=self.synthesize_max_tokens,
                    temperature=self.synthesize_temperature,
                    top_p=self.synthesize_top_p,
                ),
                llm_client=self.navigator_agent_llm_client,
                tokenizer=self.navigator_agent_tokenizer,
                use_chat_template=self.navigator_agent_use_chat_template,
            )
            phase3_ms = self._ns_to_ms(self._now_ns() - phase3_started_ns)
            batch_phase_totals["phase3_finalize_ms"] += phase3_ms
            result["timing"]["phase3_finalize_ms"] += phase3_ms

            final_output = final_outputs[0] if final_outputs else ""
            result["navigator_agent_prompts"].append(final_output)
            parsed = self._parse_answer(final_output)
            if parsed:
                result["final_answer"] = parsed
            elif final_output:
                result["final_answer"] = final_output.strip()
            else:
                result["final_answer"] = ""

        for result in results:
            phase_sum = (
                result["timing"]["phase1_navigator_ms"]
                + result["timing"]["phase2_retrieval_ms"]
                + result["timing"]["phase3_finalize_ms"]
            )
            result["timing"]["total_ms"] = phase_sum
            result["timing"]["executed_iterations"] = result["executed_iterations"]

        self.latest_batch_timing = {
            "num_questions": len(questions),
            "max_iterations": max_iterations,
            "total_ms": self._ns_to_ms(self._now_ns() - run_batch_started_ns),
            "phase_totals_ms": batch_phase_totals,
            "iteration_timings": iteration_timings,
        }

        return results
