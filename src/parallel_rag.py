import hashlib
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, TypedDict

from src.clients import (BatchSearchDocs, OpenAIClient, RetrieverClient,
                         RetrieverDocument)
from src.prompted_generation_base import (PromptedGenerationBase,
                                          build_retriever_client_from_args,
                                          parse_stop_tokens)

THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>",
                               flags=re.DOTALL | re.IGNORECASE)
SEARCH_QUERIES_TAG_PATTERN = re.compile(r"<search_queries>(.*?)</search_queries>",
                                        flags=re.DOTALL | re.IGNORECASE)
SEARCH_TAG_PATTERN = re.compile(r"<search>(.*?)</search>",
                                flags=re.DOTALL | re.IGNORECASE)
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>",
                                flags=re.DOTALL | re.IGNORECASE)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


class PooledDocument(TypedDict):
    doc_id: str
    contents: str
    source_query_ids: List[str]
    source_queries: List[str]


class ParallelRAGResult(TypedDict):
    query: str
    max_iterations: int
    executed_iterations: int
    navigator_agent_prompts: List[str]
    navigator_agent_thoughts: List[str]
    search_queries: List[List[str]]
    retrieved_docs: List[BatchSearchDocs]
    pooled_docs: List[List[PooledDocument]]
    refine_prompts: List[List[str]]
    global_refinements: List[str]
    final_answer: str
    timing: Dict[str, Any]


class ParallelRAG(PromptedGenerationBase):
    def __init__(self,
                 retriever: RetrieverClient,
                 navigator_agent_llm_client: OpenAIClient,
                 global_refine_agent_llm_client: OpenAIClient,
                 docs_per_query: int = 5,
                 navigator_agent_max_tokens: int = 512,
                 navigator_agent_temperature: float = 0.6,
                 navigator_agent_top_p: float = 0.9,
                 global_refine_agent_max_tokens: int = 1024,
                 global_refine_agent_temperature: float = 0.8,
                 global_refine_agent_top_p: float = 0.9,
                 synthesize_max_tokens: int = 768,
                 synthesize_temperature: float = 0.3,
                 synthesize_top_p: float = 0.9,
                 stop_tokens: List[str] | None = None,
                 navigator_agent_use_chat_template: bool = False,
                 navigator_agent_tokenizer: Any = None,
                 global_refine_agent_use_chat_template: bool = False,
                 global_refine_agent_tokenizer: Any = None):
        super().__init__(llm_client=navigator_agent_llm_client,
                         generation_max_tokens=synthesize_max_tokens,
                         generation_temperature=synthesize_temperature,
                         generation_top_p=synthesize_top_p,
                         stop_tokens=stop_tokens,
                         use_chat_template=navigator_agent_use_chat_template,
                         tokenizer=navigator_agent_tokenizer)
        self.retriever = retriever
        self.navigator_agent_llm_client = navigator_agent_llm_client
        self.global_refine_agent_llm_client = global_refine_agent_llm_client
        self.docs_per_query = max(1, docs_per_query)
        self.navigator_agent_max_tokens = navigator_agent_max_tokens
        self.navigator_agent_temperature = navigator_agent_temperature
        self.navigator_agent_top_p = navigator_agent_top_p
        self.global_refine_agent_max_tokens = global_refine_agent_max_tokens
        self.global_refine_agent_temperature = global_refine_agent_temperature
        self.global_refine_agent_top_p = global_refine_agent_top_p
        self.synthesize_max_tokens = synthesize_max_tokens
        self.synthesize_temperature = synthesize_temperature
        self.synthesize_top_p = synthesize_top_p
        self.navigator_agent_use_chat_template = navigator_agent_use_chat_template
        self.navigator_agent_tokenizer = navigator_agent_tokenizer
        self.global_refine_agent_use_chat_template = global_refine_agent_use_chat_template
        self.global_refine_agent_tokenizer = global_refine_agent_tokenizer

        if self.global_refine_agent_use_chat_template and self.global_refine_agent_tokenizer is None:
            raise ValueError(
                "global_refine_agent_tokenizer is required when global_refine_agent_use_chat_template=True")

        self.latest_batch_timing: Dict[str, Any] = {}

    @classmethod
    def from_args(cls, args) -> "ParallelRAG":
        navigator_agent_use_chat_template = _coerce_bool(
            _first_not_none(getattr(args,
                                    "navigator_agent_use_chat_template",
                                    None),
                            getattr(args, "shared_use_chat_template", None),
                            getattr(args, "use_chat_template", None)),
            default=False)
        global_refine_agent_use_chat_template = _coerce_bool(
            getattr(args, "global_refine_agent_use_chat_template", None),
            default=navigator_agent_use_chat_template)

        navigator_agent_model = _first_not_none(getattr(args,
                                                        "navigator_agent_model",
                                                        None),
                                                getattr(args,
                                                        "shared_model",
                                                        None),
                                                getattr(args,
                                                        "model",
                                                        None),
                                                getattr(args,
                                                        "llm_model",
                                                        None),
                                                "Qwen3-14B")
        global_refine_agent_model = _first_not_none(
            getattr(args, "global_refine_agent_model", None),
            getattr(args, "shared_model", None),
            getattr(args, "model", None),
            getattr(args, "llm_model", None),
            navigator_agent_model)

        navigator_agent_model_path = _first_not_none(
            getattr(args, "navigator_agent_model_path", None),
            getattr(args, "shared_model_path", None),
            getattr(args, "model_path", None),
            navigator_agent_model)
        global_refine_agent_model_path = _first_not_none(
            getattr(args, "global_refine_agent_model_path", None),
            getattr(args, "shared_model_path", None),
            getattr(args, "model_path", None),
            global_refine_agent_model)

        navigator_agent_tokenizer = _first_not_none(
            getattr(args, "navigator_agent_tokenizer", None),
            getattr(args, "shared_tokenizer", None))
        if navigator_agent_use_chat_template and navigator_agent_tokenizer is None:
            navigator_agent_tokenizer = _load_tokenizer(
                navigator_agent_model_path)

        global_refine_agent_tokenizer = _first_not_none(
            getattr(args, "global_refine_agent_tokenizer", None),
            getattr(args, "shared_tokenizer", None))
        if global_refine_agent_use_chat_template and global_refine_agent_tokenizer is None:
            global_refine_agent_tokenizer = _load_tokenizer(
                global_refine_agent_model_path)

        retriever_client = build_retriever_client_from_args(args)
        navigator_agent_llm_client = OpenAIClient(
            base_url=_first_not_none(getattr(args,
                                             "navigator_agent_openai_base_url",
                                             None),
                                     getattr(args,
                                             "shared_openai_base_url",
                                             None),
                                     getattr(args,
                                             "openai_base_url",
                                             None),
                                     "http://127.0.0.1:8001"),
            model=navigator_agent_model,
            api_key=_first_not_none(getattr(args,
                                            "navigator_agent_openai_api_key",
                                            None),
                                    getattr(args,
                                            "shared_openai_api_key",
                                            None),
                                    getattr(args, "openai_api_key", None),
                                    "TEST"),
            timeout=_first_not_none(getattr(args,
                                            "navigator_agent_llm_timeout",
                                            None),
                                    getattr(args,
                                            "shared_llm_timeout",
                                            None),
                                    getattr(args, "llm_timeout", None)),
            use_chat_template=navigator_agent_use_chat_template)
        global_refine_agent_llm_client = OpenAIClient(
            base_url=_first_not_none(getattr(args,
                                             "global_refine_agent_openai_base_url",
                                             None),
                                     getattr(args,
                                             "shared_openai_base_url",
                                             None),
                                     getattr(args,
                                             "openai_base_url",
                                             None),
                                     "http://127.0.0.1:8001"),
            model=global_refine_agent_model,
            api_key=_first_not_none(getattr(args,
                                            "global_refine_agent_openai_api_key",
                                            None),
                                    getattr(args,
                                            "shared_openai_api_key",
                                            None),
                                    getattr(args, "openai_api_key", None),
                                    "TEST"),
            timeout=_first_not_none(getattr(args,
                                            "global_refine_agent_llm_timeout",
                                            None),
                                    getattr(args,
                                            "shared_llm_timeout",
                                            None),
                                    getattr(args, "llm_timeout", None)),
            use_chat_template=global_refine_agent_use_chat_template)

        docs_per_query = int(getattr(args, "docs_per_query", 5))
        navigator_agent_max_tokens = int(
            getattr(args, "navigator_agent_max_tokens", 256))
        navigator_agent_temperature = float(
            getattr(args, "navigator_agent_temperature", 0.6))
        navigator_agent_top_p = float(
            getattr(args, "navigator_agent_top_p", 0.9))

        global_refine_agent_max_tokens = int(
            getattr(args,
                    "global_refine_agent_max_tokens",
                    getattr(args, "refine_max_tokens", 1024)))
        global_refine_agent_temperature = float(
            getattr(args,
                    "global_refine_agent_temperature",
                    getattr(args, "refine_temperature", 0.8)))
        global_refine_agent_top_p = float(
            getattr(args,
                    "global_refine_agent_top_p",
                    getattr(args, "refine_top_p", 0.9)))

        synthesize_max_tokens = int(
            getattr(args, "synthesize_max_tokens", 768))
        synthesize_temperature = float(
            getattr(args, "synthesize_temperature", 0.3))
        synthesize_top_p = float(getattr(args, "synthesize_top_p", 0.9))

        return cls(retriever=retriever_client,
                   navigator_agent_llm_client=navigator_agent_llm_client,
                   global_refine_agent_llm_client=global_refine_agent_llm_client,
                   docs_per_query=docs_per_query,
                   navigator_agent_max_tokens=navigator_agent_max_tokens,
                   navigator_agent_temperature=navigator_agent_temperature,
                   navigator_agent_top_p=navigator_agent_top_p,
                   global_refine_agent_max_tokens=global_refine_agent_max_tokens,
                   global_refine_agent_temperature=global_refine_agent_temperature,
                   global_refine_agent_top_p=global_refine_agent_top_p,
                   synthesize_max_tokens=synthesize_max_tokens,
                   synthesize_temperature=synthesize_temperature,
                   synthesize_top_p=synthesize_top_p,
                   stop_tokens=parse_stop_tokens(args),
                   navigator_agent_use_chat_template=navigator_agent_use_chat_template,
                   navigator_agent_tokenizer=navigator_agent_tokenizer,
                   global_refine_agent_use_chat_template=global_refine_agent_use_chat_template,
                   global_refine_agent_tokenizer=global_refine_agent_tokenizer)

    def _make_config(self, max_tokens: int, temperature: float,
                     top_p: float) -> SimpleNamespace:
        return SimpleNamespace(max_completion_length=max_tokens,
                               temperature=temperature,
                               top_p=top_p,
                               vllm_n=1)

    def _format_prompt_by_template(self,
                                   system_prompt: str,
                                   user_prompt: str,
                                   use_chat_template: bool) -> Any:
        if not use_chat_template:
            return f"{system_prompt}\n\n{user_prompt}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _extract_searches(self, text: str) -> List[str]:
        searches: List[str] = []
        search_query_blocks = SEARCH_QUERIES_TAG_PATTERN.findall(text)
        scopes = search_query_blocks if search_query_blocks else [text]

        for scope in scopes:
            for matched in SEARCH_TAG_PATTERN.findall(scope):
                query = matched.strip()
                if query and query not in searches:
                    searches.append(query)
        return searches

    def _parse_answer(self, text: str) -> str:
        return self._extract_last_tag(ANSWER_TAG_PATTERN, text)

    def _extract_think(self, text: str) -> str:
        extracted = self._extract_last_tag(THINK_TAG_PATTERN, text)
        return extracted if extracted else text.strip()

    def _generate_text_batch(self, prompts: List[Any],
                             config: SimpleNamespace,
                             llm_client: OpenAIClient,
                             tokenizer: Any = None,
                             use_chat_template: bool | None = None) -> List[str]:
        if not prompts:
            return []

        active_use_chat_template = self.use_chat_template if use_chat_template is None else bool(
            use_chat_template)

        if not active_use_chat_template:
            prompt_texts = [self._prompt_to_text(p) for p in prompts]
            return llm_client.generate_text(prompt_texts,
                                            config,
                                            self.stop_tokens)

        return llm_client.generate_text(prompts,
                                        config,
                                        self.stop_tokens,
                                        tokenizer=tokenizer)

    def _build_navigator_agent_prompt(self,
                                      question: str,
                                      historical_refinements: List[str],
                                      use_chat_template: bool) -> Any:
        history_block = "\n\n".join([
            f"R_{idx + 1}:\n{report}"
            for idx, report in enumerate(historical_refinements)
        ]) if historical_refinements else "None"

        system_prompt = (
            "You are a navigator_agent in a multi-round retrieval reasoning system. "
            "You receive the original question and historical refined reports. "
            "First think inside <think>...</think>. "
            "If information is sufficient, output final concise answer inside <answer>...</answer>. "
            "If information is insufficient, output one or multiple concrete web search queries wrapped inside "
            "<search_queries>...</search_queries>, and wrap each query with <search>...</search>. "
        )
        user_prompt = "\n\n".join([
            self._format_external_context("Original Question", question),
            self._format_external_context("Historical Refined Information",
                                          history_block),
            "Decide whether to answer now or output multiple queries in <search_queries> with each query in <search>.",
        ])
        return self._format_prompt_by_template(system_prompt,
                                               user_prompt,
                                               use_chat_template)

    def _build_global_refine_agent_prompt(self,
                                          question: str,
                                          navigator_agent_think: str,
                                          search_queries: List[str],
                                          pooled_docs: List[PooledDocument],
                                          use_chat_template: bool) -> Any:
        query_block = "\n".join([
            f"- Query {idx + 1}: {query}"
            for idx, query in enumerate(search_queries)
        ]) if search_queries else "None"

        docs_block = "\n\n".join([
            "\n".join([
                f"DocID: {doc['doc_id']}",
                f"Source Query IDs: {' | '.join(doc['source_query_ids'])}",
                f"Source Queries: {' | '.join(doc['source_queries'])}",
                f"Content: {doc['contents']}",
            ])
            for doc in pooled_docs
        ]) if pooled_docs else "No documents in the pooled knowledge base."

        system_prompt = (
            "You are the Global Refine Agent.\n"
            "You are tasked with reading and analyzing document based on the following inputs: the original question, search queries, and a global document pool with provenance metadata.\n"
            "Your objective is to extract relevant and helpful information for Current Search Query from the Searched document pool and seamlessly integrate this information into the Previous Reasoning Steps to continue reasoning for the original question.  Guidelines:\n"
            "1. Analyze the Searched document pool:\n"
            "  - Carefully review the content of each searched web page.\n"
            "  - Identify factual information that is relevant to the Current Search Query and can aid in the reasoning process for the original question.\n"
            "2. Extract Relevant Information:\n"
            "  - Select the information from the Searched document pool that  directly contributes to advancing the Previous Reasoning  Steps.\n"
            "  - Ensure that the extracted information is accurate and relevant.\n"
            "3. Output Format: "
            "  - If the document pool provide helpful information for current search query: Present the information beginning with `Final Information` as shown below.\n"
            "Final Information[Helpful information here]\n"
            "  - If the document pool do not provide any helpful information for current search query: Output the following text.\n"
            "Final Information No helpful information found.\n\n"
            "Inputs:"
        )
        user_prompt = "\n\n".join([
            self._format_external_context("Original Question", question),
            self._format_external_context("Previous Navigator Agent Thinking",
                                          navigator_agent_think),
            self._format_external_context("Search Queries", query_block),
            self._format_external_context("Global Document Pool", docs_block),
            "Now you should analyze each web page and find  helpful information based on the original question, this round's direction set, the concrete queries, and a global document pool with provenance metadata.",

        ])
        return self._format_prompt_by_template(system_prompt,
                                               user_prompt,
                                               use_chat_template)

    def _build_refinement_appendix(self, report: str) -> str:
        return (
            "\n<information>\n"
            f"{report}"
            "\n</information>\n"
        )

    def _build_final_answer_prompt(self,
                                   question: str,
                                   historical_refinements: List[str],
                                   use_chat_template: bool) -> Any:
        history_block = "\n\n".join([
            f"R_{idx + 1}:\n{report}"
            for idx, report in enumerate(historical_refinements)
        ]) if historical_refinements else "None"
        system_prompt = (
            "You are the navigator_agent. "
            "Use historical refined reports to provide the best final answer now. "
            "Output only concise final answer inside <answer>...</answer>."
        )
        user_prompt = "\n\n".join([
            self._format_external_context("Original Question", question),
            self._format_external_context("Historical Global Refinements",
                                          history_block),
        ])
        return self._format_prompt_by_template(system_prompt,
                                               user_prompt,
                                               use_chat_template)

    def _make_doc_key(self, doc: RetrieverDocument) -> str:
        doc_id = str(doc.get("id", "")).strip()
        if doc_id:
            return doc_id
        contents = str(doc.get("contents", ""))
        return hashlib.sha1(contents.encode("utf-8")).hexdigest()

    def _pool_documents(self,
                        search_queries: List[str],
                        docs_map: dict[tuple[int, int], List[RetrieverDocument]],
                        sample_i: int) -> List[PooledDocument]:
        pooled_by_key: dict[str, PooledDocument] = {}
        for query_id, query in enumerate(search_queries, start=1):
            docs = docs_map.get((sample_i, query_id), [])
            for doc in docs[:self.docs_per_query]:
                key = self._make_doc_key(doc)
                contents = str(doc.get("contents", "")).strip()
                if key not in pooled_by_key:
                    pooled_by_key[key] = {
                        "doc_id": str(doc.get("id", "")).strip() or key,
                        "contents": contents,
                        "source_query_ids": [str(query_id)],
                        "source_queries": [query],
                    }
                    continue

                pooled_doc = pooled_by_key[key]
                if str(query_id) not in pooled_doc["source_query_ids"]:
                    pooled_doc["source_query_ids"].append(str(query_id))
                if query not in pooled_doc["source_queries"]:
                    pooled_doc["source_queries"].append(query)

        return list(pooled_by_key.values())

    @staticmethod
    def _now_ns() -> int:
        return time.perf_counter_ns()

    @staticmethod
    def _ns_to_ms(duration_ns: int) -> float:
        return duration_ns / 1_000_000.0

    def _init_batch_runtime(self,
                            questions: List[str],
                            max_iterations: int) -> tuple[List[bool], List[ParallelRAGResult], List[List[str]]]:
        completed = [False] * len(questions)

        results: List[ParallelRAGResult] = [{
            "query": q,
            "max_iterations": max_iterations,
            "executed_iterations": 0,
            "navigator_agent_prompts": [],
            "navigator_agent_thoughts": [],
            "search_queries": [],
            "retrieved_docs": [],
            "pooled_docs": [],
            "refine_prompts": [],
            "global_refinements": [],
            "final_answer": "",
            "timing": {
                "phase1_navigator_ms": 0.0,
                "phase2_retrieval_ms": 0.0,
                "phase2_pooling_ms": 0.0,
                "phase3_refine_ms": 0.0,
                "phase4_synthesize_ms": 0.0,
                "total_ms": 0.0,
                "executed_iterations": 0,
            },
        } for q in questions]

        refinement_histories: List[List[str]] = [[] for _ in questions]
        return completed, results, refinement_histories

    def run(self, question: str, max_iterations: int = 4) -> ParallelRAGResult:
        return self.run_batch([question], max_iterations=max_iterations)[0]

    def run_batch(self,
                  questions: List[str],
                  max_iterations: int = 5) -> List[ParallelRAGResult]:
        if not questions:
            return []

        run_batch_started_ns = self._now_ns()
        batch_phase_totals: Dict[str, float] = {
            "phase1_navigator_ms": 0.0,
            "phase2_retrieval_ms": 0.0,
            "phase2_pooling_ms": 0.0,
            "phase3_refine_ms": 0.0,
            "phase4_synthesize_ms": 0.0,
        }
        iteration_timings: List[Dict[str, Any]] = []

        completed, results, refinement_histories = self._init_batch_runtime(
            questions=questions,
            max_iterations=max_iterations)

        navigator_agent_config = self._make_config(
            max_tokens=self.navigator_agent_max_tokens,
            temperature=self.navigator_agent_temperature,
            top_p=self.navigator_agent_top_p)
        global_refine_agent_config = self._make_config(
            max_tokens=self.global_refine_agent_max_tokens,
            temperature=self.global_refine_agent_temperature,
            top_p=self.global_refine_agent_top_p)

        for iteration_idx in range(max_iterations):
            iteration_started_ns = self._now_ns()
            iteration_timing: Dict[str, Any] = {
                "iteration": iteration_idx + 1,
                "active_samples": 0,
                "search_query_count": 0,
                "refine_prompt_count": 0,
                "phase1_navigator_ms": 0.0,
                "phase2_retrieval_ms": 0.0,
                "phase2_pooling_ms": 0.0,
                "phase3_refine_ms": 0.0,
                "iteration_total_ms": 0.0,
            }

            active_indices = [i for i, c in enumerate(completed) if not c]
            if not active_indices:
                break

            iteration_timing["active_samples"] = len(active_indices)

            active_prompts = [
                self._build_navigator_agent_prompt(
                    question=questions[i],
                    historical_refinements=refinement_histories[i],
                    use_chat_template=self.navigator_agent_use_chat_template)
                for i in active_indices
            ]

            phase1_started_ns = self._now_ns()
            navigator_outputs = self._generate_text_batch(
                active_prompts,
                navigator_agent_config,
                llm_client=self.navigator_agent_llm_client,
                tokenizer=self.navigator_agent_tokenizer,
                use_chat_template=self.navigator_agent_use_chat_template)
            phase1_ms = self._ns_to_ms(self._now_ns() - phase1_started_ns)
            batch_phase_totals["phase1_navigator_ms"] += phase1_ms
            iteration_timing["phase1_navigator_ms"] = phase1_ms
            phase1_share = phase1_ms / len(active_indices)
            for sample_i in active_indices:
                results[sample_i]["timing"]["phase1_navigator_ms"] += phase1_share

            sample_search_queries: List[List[str]] = [
                [] for _ in range(len(questions))
            ]
            flat_queries: List[str] = []
            query_meta: List[tuple[int, int]] = []

            for local_i, sample_i in enumerate(active_indices):
                navigator_output = navigator_outputs[local_i]
                if not results[sample_i]["navigator_agent_prompts"]:
                    results[sample_i]["navigator_agent_prompts"].append(
                        self._prompt_to_text(active_prompts[local_i]))
                results[sample_i]["navigator_agent_prompts"].append(
                    navigator_output)
                results[sample_i]["executed_iterations"] += 1

                navigator_think = self._extract_think(navigator_output)
                results[sample_i]["navigator_agent_thoughts"].append(
                    navigator_think)

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
                    self._now_ns() - iteration_started_ns)
                iteration_timings.append(iteration_timing)
                continue

            phase2_retrieval_started_ns = self._now_ns()
            flat_docs = self.retriever.batch_search(flat_queries)
            phase2_retrieval_ms = self._ns_to_ms(
                self._now_ns() - phase2_retrieval_started_ns)
            batch_phase_totals["phase2_retrieval_ms"] += phase2_retrieval_ms
            iteration_timing["phase2_retrieval_ms"] = phase2_retrieval_ms

            retrieval_samples = sorted(
                {sample_i for sample_i, _ in query_meta})
            if retrieval_samples:
                retrieval_share = phase2_retrieval_ms / len(retrieval_samples)
                for sample_i in retrieval_samples:
                    results[sample_i]["timing"]["phase2_retrieval_ms"] += retrieval_share

            docs_map: dict[tuple[int, int], List[RetrieverDocument]] = {}
            for idx, docs in enumerate(flat_docs):
                docs_map[query_meta[idx]] = docs

            refine_prompts: List[Any] = []
            refine_meta: List[tuple[int, str]] = []

            phase2_pooling_started_ns = self._now_ns()
            for sample_i in active_indices:
                search_queries = sample_search_queries[sample_i]
                if not search_queries:
                    continue

                retrieved_for_sample: BatchSearchDocs = [
                    docs_map.get((sample_i, query_id), [])[
                        :self.docs_per_query]
                    for query_id, _ in enumerate(search_queries, start=1)
                ]
                results[sample_i]["retrieved_docs"].append(
                    retrieved_for_sample)

                pooled_docs = self._pool_documents(search_queries,
                                                   docs_map,
                                                   sample_i)
                results[sample_i]["pooled_docs"].append(pooled_docs)

                refine_prompt = self._build_global_refine_agent_prompt(
                    question=questions[sample_i],
                    navigator_agent_think=results[sample_i]["navigator_agent_thoughts"][-1],
                    search_queries=search_queries,
                    pooled_docs=pooled_docs,
                    use_chat_template=self.global_refine_agent_use_chat_template)
                refine_prompts.append(refine_prompt)
                refine_meta.append(
                    (sample_i, self._prompt_to_text(refine_prompt)))

            phase2_pooling_ms = self._ns_to_ms(self._now_ns() -
                                               phase2_pooling_started_ns)
            batch_phase_totals["phase2_pooling_ms"] += phase2_pooling_ms
            iteration_timing["phase2_pooling_ms"] = phase2_pooling_ms

            pooling_samples = [sample_i for sample_i, _ in refine_meta]
            if pooling_samples:
                pooling_share = phase2_pooling_ms / len(pooling_samples)
                for sample_i in pooling_samples:
                    results[sample_i]["timing"]["phase2_pooling_ms"] += pooling_share

            iteration_timing["refine_prompt_count"] = len(refine_prompts)
            if not refine_prompts:
                iteration_timing["iteration_total_ms"] = self._ns_to_ms(
                    self._now_ns() - iteration_started_ns)
                iteration_timings.append(iteration_timing)
                continue

            phase3_started_ns = self._now_ns()
            refine_outputs = self._generate_text_batch(
                refine_prompts,
                global_refine_agent_config,
                llm_client=self.global_refine_agent_llm_client,
                tokenizer=self.global_refine_agent_tokenizer,
                use_chat_template=self.global_refine_agent_use_chat_template)
            phase3_ms = self._ns_to_ms(self._now_ns() - phase3_started_ns)
            batch_phase_totals["phase3_refine_ms"] += phase3_ms
            iteration_timing["phase3_refine_ms"] = phase3_ms

            refine_samples = [sample_i for sample_i, _ in refine_meta]
            if refine_samples:
                refine_share = phase3_ms / len(refine_samples)
                for sample_i in refine_samples:
                    results[sample_i]["timing"]["phase3_refine_ms"] += refine_share

            for idx, refine_output in enumerate(refine_outputs):
                sample_i, refine_prompt_text = refine_meta[idx]
                results[sample_i]["refine_prompts"].append(
                    [refine_prompt_text])
                results[sample_i]["global_refinements"].append(refine_output)
                refinement_histories[sample_i].append(refine_output)
                results[sample_i]["navigator_agent_prompts"].append(
                    self._build_refinement_appendix(refine_output))

            iteration_timing["iteration_total_ms"] = self._ns_to_ms(
                self._now_ns() - iteration_started_ns)
            iteration_timings.append(iteration_timing)

        for idx, result in enumerate(results):
            if result["final_answer"]:
                continue

            final_prompt = self._build_final_answer_prompt(
                question=questions[idx],
                historical_refinements=refinement_histories[idx],
                use_chat_template=self.navigator_agent_use_chat_template)
            if not result["navigator_agent_prompts"]:
                result["navigator_agent_prompts"].append(
                    self._prompt_to_text(final_prompt))

            phase4_started_ns = self._now_ns()
            final_outputs = self._generate_text_batch(
                [final_prompt],
                self._make_config(max_tokens=self.synthesize_max_tokens,
                                  temperature=self.synthesize_temperature,
                                  top_p=self.synthesize_top_p),
                llm_client=self.navigator_agent_llm_client,
                tokenizer=self.navigator_agent_tokenizer,
                use_chat_template=self.navigator_agent_use_chat_template)
            phase4_ms = self._ns_to_ms(self._now_ns() - phase4_started_ns)
            batch_phase_totals["phase4_synthesize_ms"] += phase4_ms
            result["timing"]["phase4_synthesize_ms"] += phase4_ms

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
                + result["timing"]["phase2_pooling_ms"]
                + result["timing"]["phase3_refine_ms"]
                + result["timing"]["phase4_synthesize_ms"]
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
