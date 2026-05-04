import hashlib
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, TypedDict, cast

from src.clients import (
    BatchSearchDocs,
    OpenAIClient,
    RetrieverClient,
    RetrieverDocument,
)
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
SEARCH_DIRECTIONS_PATTERN = re.compile(
    r"<search_directions>(.*?)</search_directions>", flags=re.DOTALL | re.IGNORECASE
)
DIRECTION_TAG_PATTERN = re.compile(
    r"<direction(?:\s+id=[\"']?(.*?)[\"']?)?>(.*?)</direction>",
    flags=re.DOTALL | re.IGNORECASE,
)


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


class SearchDirection(TypedDict):
    direction_id: str
    direction: str


class AgentAnswerRecord(TypedDict):
    prompt: str
    output: str
    answer: str


class PathAgentCoreRecord(TypedDict):
    path_id: int
    direction_id: str
    direction: str
    search_query: str


class PathAgentIterationRecord(PathAgentCoreRecord):
    prompt: str
    navigator_agent_think: str
    path_agent_think: str
    path_agent_output: str


class PooledDocument(TypedDict):
    doc_id: str
    contents: str
    sources: List[str]
    source_directions: List[str]
    source_queries: List[str]


class NavigatorIterationRecord(AgentAnswerRecord):
    think: str
    search_directions: List[SearchDirection]


class IterationContext(TypedDict):
    """Stores per-iteration context (path plans + pooled docs) for the navigator agent."""

    directions: List[SearchDirection]
    path_plans: List[PathAgentIterationRecord]
    pooled_docs: List[PooledDocument]


class FinalSynthesisRecord(AgentAnswerRecord):
    pass


class IterationRecord(TypedDict, total=False):
    iteration: int
    navigator: NavigatorIterationRecord
    path_agents: List[PathAgentIterationRecord]


class ParallelO1Result(TypedDict):
    query: str
    max_iterations: int
    executed_iterations: int
    iterations: List[IterationRecord]
    final_synthesis: FinalSynthesisRecord
    final_answer: str
    timing: Dict[str, Any]


class AdaptiveParallelO1NoRefine(PromptedGenerationBase):
    """Ablation variant of AdaptiveParallelO1 WITHOUT the global refine agent.

    Instead of a separate refiner, path agent outputs and pooled documents are
    fed directly into the Navigator agent as iteration context.
    """

    def __init__(
        self,
        retriever: RetrieverClient,
        navigator_agent_llm_client: OpenAIClient,
        path_agent_llm_client: OpenAIClient,
        navigator_agent_max_tokens: int = 512,
        navigator_agent_temperature: float = 0.6,
        navigator_agent_top_p: float = 0.9,
        path_agent_max_tokens: int = 512,
        path_agent_temperature: float = 0.8,
        path_agent_top_p: float = 0.95,
        synthesize_max_tokens: int = 512,
        synthesize_temperature: float = 0.3,
        synthesize_top_p: float = 0.9,
        stop_tokens: List[str] | None = None,
        navigator_agent_use_chat_template: bool = False,
        navigator_agent_tokenizer: Any = None,
        navigator_agent_enable_thinking: bool = False,
        path_agent_use_chat_template: bool = False,
        path_tokenizer: Any = None,
        path_agent_enable_thinking: bool = False,
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
        self.path_agent_llm_client = path_agent_llm_client
        self.navigator_agent_max_tokens = navigator_agent_max_tokens
        self.navigator_agent_temperature = navigator_agent_temperature
        self.navigator_agent_top_p = navigator_agent_top_p
        self.path_agent_max_tokens = path_agent_max_tokens
        self.path_agent_temperature = path_agent_temperature
        self.path_agent_top_p = path_agent_top_p
        self.synthesize_max_tokens = synthesize_max_tokens
        self.synthesize_temperature = synthesize_temperature
        self.synthesize_top_p = synthesize_top_p
        self.navigator_agent_use_chat_template = navigator_agent_use_chat_template
        self.navigator_agent_tokenizer = navigator_agent_tokenizer
        self.navigator_agent_enable_thinking = navigator_agent_enable_thinking
        self.path_agent_use_chat_template = path_agent_use_chat_template
        self.path_agent_tokenizer = path_tokenizer
        self.path_agent_enable_thinking = path_agent_enable_thinking

        if self.path_agent_use_chat_template and self.path_agent_tokenizer is None:
            raise ValueError(
                "path_tokenizer is required when path_agent_use_chat_template=True"
            )
        self.latest_batch_timing: Dict[str, Any] = {}

    @classmethod
    def from_args(cls, args) -> "AdaptiveParallelO1NoRefine":
        navigator_agent_use_chat_template = _coerce_bool(
            _first_not_none(
                getattr(args, "navigator_agent_use_chat_template", None),
                getattr(args, "shared_use_chat_template", None),
                getattr(args, "use_chat_template", None),
            ),
            default=False,
        )
        navigator_agent_enable_thinking = _coerce_bool(
            _first_not_none(
                getattr(args, "navigator_agent_enable_thinking", None),
                getattr(args, "shared_enable_thinking", None),
                getattr(args, "enable_thinking", None),
            ),
            default=False,
        )
        path_agent_use_chat_template_raw = getattr(
            args, "path_agent_use_chat_template", None
        )
        path_agent_use_chat_template = _coerce_bool(
            path_agent_use_chat_template_raw, default=navigator_agent_use_chat_template
        )
        path_agent_enable_thinking_raw = getattr(
            args, "path_agent_enable_thinking", None
        )
        path_agent_enable_thinking = _coerce_bool(
            path_agent_enable_thinking_raw,
            default=navigator_agent_enable_thinking,
        )

        navigator_agent_model = _first_not_none(
            getattr(args, "navigator_agent_model", None),
            getattr(args, "shared_model", None),
            getattr(args, "model", None),
            getattr(args, "llm_model", None),
            "Qwen3-14B",
        )
        path_agent_model = (
            getattr(args, "path_agent_model", None) or navigator_agent_model
        )

        navigator_agent_model_path = _first_not_none(
            getattr(args, "navigator_agent_model_path", None),
            getattr(args, "shared_model_path", None),
            getattr(args, "model_path", None),
            navigator_agent_model,
        )
        path_agent_model_path = (
            getattr(args, "path_agent_model_path", None) or path_agent_model
        )

        navigator_agent_tokenizer = _first_not_none(
            getattr(args, "navigator_agent_tokenizer", None),
            getattr(args, "shared_tokenizer", None),
        )
        if navigator_agent_use_chat_template and navigator_agent_tokenizer is None:
            navigator_agent_tokenizer = _load_tokenizer(navigator_agent_model_path)

        path_tokenizer = getattr(args, "path_tokenizer", None)
        if path_agent_use_chat_template and path_tokenizer is None:
            path_tokenizer = _load_tokenizer(path_agent_model_path)

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
        path_agent_llm_client = OpenAIClient(
            base_url=_first_not_none(
                getattr(args, "path_agent_openai_base_url", None),
                getattr(args, "navigator_agent_openai_base_url", None),
                getattr(args, "shared_openai_base_url", None),
                getattr(args, "openai_base_url", None),
                "http://127.0.0.1:8001",
            ),
            model=path_agent_model,
            api_key=_first_not_none(
                getattr(args, "path_agent_openai_api_key", None),
                getattr(args, "navigator_agent_openai_api_key", None),
                getattr(args, "shared_openai_api_key", None),
                getattr(args, "openai_api_key", None),
                "TEST",
            ),
            timeout=_first_not_none(
                getattr(args, "path_agent_llm_timeout", None),
                getattr(args, "navigator_agent_llm_timeout", None),
                getattr(args, "shared_llm_timeout", None),
                getattr(args, "llm_timeout", None),
            ),
            use_chat_template=path_agent_use_chat_template,
        )

        navigator_agent_max_tokens = int(
            getattr(args, "navigator_agent_max_tokens", 256)
        )
        navigator_agent_temperature = float(
            getattr(args, "navigator_agent_temperature", 0.6)
        )
        navigator_agent_top_p = float(getattr(args, "navigator_agent_top_p", 0.9))

        path_agent_max_tokens = int(getattr(args, "path_agent_max_tokens", 384))
        path_agent_temperature = float(getattr(args, "path_agent_temperature", 0.8))
        path_agent_top_p = float(getattr(args, "path_agent_top_p", 0.95))

        synthesize_max_tokens = int(getattr(args, "synthesize_max_tokens", 768))
        synthesize_temperature = float(getattr(args, "synthesize_temperature", 0.3))
        synthesize_top_p = float(getattr(args, "synthesize_top_p", 0.9))
        return cls(
            retriever=retriever_client,
            navigator_agent_llm_client=navigator_agent_llm_client,
            path_agent_llm_client=path_agent_llm_client,
            navigator_agent_max_tokens=navigator_agent_max_tokens,
            navigator_agent_temperature=navigator_agent_temperature,
            navigator_agent_top_p=navigator_agent_top_p,
            path_agent_max_tokens=path_agent_max_tokens,
            path_agent_temperature=path_agent_temperature,
            path_agent_top_p=path_agent_top_p,
            synthesize_max_tokens=synthesize_max_tokens,
            synthesize_temperature=synthesize_temperature,
            synthesize_top_p=synthesize_top_p,
            stop_tokens=parse_stop_tokens(args),
            navigator_agent_use_chat_template=navigator_agent_use_chat_template,
            navigator_agent_tokenizer=navigator_agent_tokenizer,
            navigator_agent_enable_thinking=navigator_agent_enable_thinking,
            path_agent_use_chat_template=path_agent_use_chat_template,
            path_tokenizer=path_tokenizer,
            path_agent_enable_thinking=path_agent_enable_thinking,
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

    def _extract_searches(self, text: str) -> List[str]:
        searches: List[str] = []
        for matched in SEARCH_TAG_PATTERN.findall(text):
            q = matched.strip()
            if q and q not in searches:
                searches.append(q)
        return searches

    def _generate_text_batch(
        self,
        prompts: List[Any],
        config: SimpleNamespace,
        llm_client: OpenAIClient,
        tokenizer: Any = None,
        use_chat_template: bool | None = None,
        enable_thinking: bool = False,
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
            prompts,
            config,
            self.stop_tokens,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
        )

    def _extract_think(self, text: str) -> str:
        extracted = self._extract_last_tag(THINK_TAG_PATTERN, text)
        return extracted if extracted else text.strip()

    def _parse_search_directions(self, text: str) -> List[SearchDirection]:
        block_match = SEARCH_DIRECTIONS_PATTERN.search(text)
        if not block_match:
            return []

        block = block_match.group(1)
        directions: List[SearchDirection] = []
        for idx, match in enumerate(DIRECTION_TAG_PATTERN.finditer(block), start=1):
            direction_id = (match.group(1) or str(idx)).strip() or str(idx)
            direction_text = re.sub(r"\s+", " ", match.group(2)).strip()
            if not direction_text:
                continue
            directions.append(
                {
                    "direction_id": direction_id,
                    "direction": direction_text,
                }
            )

        return directions

    # ------------------------------------------------------------------
    # Prompt builders (no-refine variant — Navigator receives raw path
    # outputs + pooled docs instead of refined reports)
    # ------------------------------------------------------------------

    def _format_iteration_contexts(
        self, iteration_contexts: List[IterationContext]
    ) -> str:
        """Format accumulated iteration contexts into a text block for the Navigator."""
        if not iteration_contexts:
            return "None"

        blocks: List[str] = []
        for iter_idx, ctx in enumerate(iteration_contexts, start=1):
            lines: List[str] = [f"--- Iteration {iter_idx} ---"]

            directions = ctx.get("directions", [])
            path_plans_by_id: Dict[str, PathAgentIterationRecord] = {}
            for plan in ctx.get("path_plans", []):
                path_plans_by_id[plan["direction_id"]] = plan

            # Per-direction: show abstract direction, concrete query, and path agent output
            for direction in directions:
                did = direction["direction_id"]
                lines.append(f"Direction {did}: {direction['direction']}")
                plan = path_plans_by_id.get(did)
                if plan:
                    lines.append(f"  Search Query: {plan['search_query']}")
                    if plan.get("path_agent_think"):
                        lines.append(f"  Path Agent Reasoning: {plan['path_agent_think']}")
                    if plan.get("path_agent_output"):
                        lines.append(
                            f"  Path Agent Output: {plan['path_agent_output']}"
                        )

            # Pooled documents
            pooled_docs = ctx.get("pooled_docs", [])
            if pooled_docs:
                lines.append("Retrieved Documents:")
                for doc in pooled_docs:
                    lines.append(f"  - DocID: {doc['doc_id']}")
                    lines.append(
                        f"    Source Directions: {', '.join(doc['source_directions'])}"
                    )
                    lines.append(
                        f"    Source Queries: {' | '.join(doc['source_queries'])}"
                    )
                    lines.append(f"    Content: {doc['contents']}")
                    lines.append("")

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    def _build_navigator_agent_prompt(
        self,
        question: str,
        iteration_contexts: List[IterationContext],
        use_chat_template: bool,
    ) -> Any:
        """Build Navigator prompt — now receives raw iteration contexts (no refiner)."""
        history_block = self._format_iteration_contexts(iteration_contexts)

        system_prompt = (
            "You are a navigator agent in a multi-stage retrieval reasoning system to answer the user's question. "
            "You receive the original question and information from previous retrieval iterations, "
            "including the abstract search directions, concrete search queries from path agents, "
            "and the retrieved documents with their provenance metadata. "
            "You must first think globally inside <think>...</think>. "
            "If the available information is sufficient, output the final concise short answer inside <answer>...</answer>. "
            "If you lack of some external information, you can output a dynamic set of abstract retrieval directions inside <search_directions>...</search_directions>. "
            'Each direction must be wrapped in <direction id="k">...</direction>. '
            "You may produce as many directions as needed for this round."
        )
        user_prompt = "\n\n".join(
            [
                self._format_external_context("Original Question", question),
                self._format_external_context(
                    "Previous Iteration Information", history_block
                ),
                "Decide whether to answer now or propose the next retrieval directions.",
            ]
        )
        return self._format_prompt_by_template(
            system_prompt, user_prompt, use_chat_template
        )

    def _build_path_agent_prompt(
        self,
        original_question: str,
        navigator_agent_think: str,
        direction: SearchDirection,
        use_chat_template: bool,
    ) -> Any:
        system_prompt = (
            "You are a helpful assistant that help Navigator Agent to answer user's question. "
            "You will receive the original question, the Navigator Agent's current thoughts, and one assigned retrieval direction by Navigator Agent. "
            "Your task is to convert the abstract retrieval direction into one concrete search-engine-friendly query. "
            "You must reason locally inside <think>...</think> and then output exactly one search query inside <search>...</search>. "
            "The Inputs:"
        )
        user_prompt = "\n\n".join(
            [
                self._format_external_context("Original Question", original_question),
                self._format_external_context(
                    "navigator_agent Thinking", navigator_agent_think
                ),
                self._format_external_context(
                    f"Assigned Direction {direction['direction_id']}",
                    direction["direction"],
                ),
                "Now think and put your search query in <search>...</search>",
            ]
        )
        return self._format_prompt_by_template(
            system_prompt, user_prompt, use_chat_template
        )

    def _build_final_answer_prompt(
        self,
        question: str,
        iteration_contexts: List[IterationContext],
        use_chat_template: bool,
    ) -> Any:
        """Build final-answer prompt using accumulated iteration contexts (no refiner)."""
        history_block = self._format_iteration_contexts(iteration_contexts)

        system_prompt = (
            "You are the navigator_agent Agent. "
            "Use the historical retrieval iteration information to provide the best final answer now. "
            "Output only the concise final answer inside <answer>...</answer>."
        )
        user_prompt = "\n\n".join(
            [
                self._format_external_context("Original Question", question),
                self._format_external_context(
                    "Historical Retrieval Iterations", history_block
                ),
            ]
        )
        return self._format_prompt_by_template(
            system_prompt, user_prompt, use_chat_template
        )

    # ------------------------------------------------------------------
    # Document pooling (unchanged from original)
    # ------------------------------------------------------------------

    def _make_doc_key(self, doc: RetrieverDocument) -> str:
        doc_id = str(doc.get("id", "")).strip()
        if doc_id:
            return doc_id
        contents = str(doc.get("contents", ""))
        return hashlib.sha1(contents.encode("utf-8")).hexdigest()

    def _pool_documents(
        self,
        path_plans: List[PathAgentIterationRecord],
        docs_map: dict[tuple[int, int], List[RetrieverDocument]],
        sample_i: int,
    ) -> List[PooledDocument]:
        pooled_by_key: dict[str, PooledDocument] = {}
        for plan in path_plans:
            docs = docs_map.get((sample_i, plan["path_id"]), [])
            for doc in docs:
                key = self._make_doc_key(doc)
                contents = str(doc.get("contents", "")).strip()
                if key not in pooled_by_key:
                    pooled_by_key[key] = {
                        "doc_id": str(doc.get("id", "")).strip() or key,
                        "contents": contents,
                        "sources": [plan["direction_id"]],
                        "source_directions": [plan["direction"]],
                        "source_queries": [plan["search_query"]],
                    }
                    continue

                pooled_doc = pooled_by_key[key]
                if plan["direction_id"] not in pooled_doc["sources"]:
                    pooled_doc["sources"].append(plan["direction_id"])
                if plan["direction"] not in pooled_doc["source_directions"]:
                    pooled_doc["source_directions"].append(plan["direction"])
                if plan["search_query"] not in pooled_doc["source_queries"]:
                    pooled_doc["source_queries"].append(plan["search_query"])

        return list(pooled_by_key.values())

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    def _parse_answer(self, text: str) -> str:
        return self._extract_last_tag(ANSWER_TAG_PATTERN, text)

    @staticmethod
    def _now_ns() -> int:
        return time.perf_counter_ns()

    @staticmethod
    def _ns_to_ms(duration_ns: int) -> float:
        return duration_ns / 1_000_000.0

    def _init_batch_runtime(
        self, questions: List[str], max_iterations: int
    ) -> tuple[
        List[bool],
        List[ParallelO1Result],
        List[List[IterationContext]],
    ]:
        """Initialize per-sample runtime states for batched execution (no-refine)."""
        completed = [False] * len(questions)

        results: List[ParallelO1Result] = [
            {
                "query": q,
                "max_iterations": max_iterations,
                "executed_iterations": 0,
                "iterations": [],
                "final_synthesis": {
                    "prompt": "",
                    "output": "",
                    "answer": "",
                },
                "final_answer": "",
                "timing": {
                    "phase1_navigator_ms": 0.0,
                    "phase2_path_ms": 0.0,
                    "phase3_retrieval_ms": 0.0,
                    "phase3_pooling_ms": 0.0,
                    "phase5_synthesize_ms": 0.0,
                    "total_ms": 0.0,
                    "executed_iterations": 0,
                },
            }
            for q in questions
        ]

        # Accumulate per-iteration context (path plans + pooled docs) for each sample.
        iteration_contexts: List[List[IterationContext]] = [
            [] for _ in questions
        ]
        return completed, results, iteration_contexts

    def run(self, question: str, max_iterations: int = 4) -> ParallelO1Result:
        return self.run_batch([question], max_iterations=max_iterations)[0]

    def run_batch(
        self, questions: List[str], max_iterations: int = 4
    ) -> List[ParallelO1Result]:
        # -----------------------------
        # Phase 0: Runtime initialization
        # -----------------------------
        if not questions:
            return []

        run_batch_started_ns = self._now_ns()
        batch_phase_totals: Dict[str, float] = {
            "phase1_navigator_ms": 0.0,
            "phase2_path_ms": 0.0,
            "phase3_retrieval_ms": 0.0,
            "phase3_pooling_ms": 0.0,
            "phase5_synthesize_ms": 0.0,
        }
        iteration_timings: List[Dict[str, Any]] = []

        completed, results, iteration_contexts = self._init_batch_runtime(
            questions=questions, max_iterations=max_iterations
        )
        latest_navigator_thinks = ["" for _ in questions]

        # Static generation configs used in each stage.
        navigator_agent_config = self._make_config(
            max_tokens=self.navigator_agent_max_tokens,
            temperature=self.navigator_agent_temperature,
            top_p=self.navigator_agent_top_p,
        )
        path_config = self._make_config(
            max_tokens=self.path_agent_max_tokens,
            temperature=self.path_agent_temperature,
            top_p=self.path_agent_top_p,
        )

        for iteration_idx in range(max_iterations):
            iteration_started_ns = self._now_ns()
            iteration_timing: Dict[str, Any] = {
                "iteration": iteration_idx + 1,
                "active_samples": 0,
                "path_prompt_count": 0,
                "retrieval_query_count": 0,
                "phase1_navigator_ms": 0.0,
                "phase2_path_ms": 0.0,
                "phase3_retrieval_ms": 0.0,
                "phase3_pooling_ms": 0.0,
                "iteration_total_ms": 0.0,
            }

            # -----------------------------
            # Phase 1: Navigator planning
            # -----------------------------
            active_indices = [i for i, c in enumerate(completed) if not c]
            if not active_indices:
                break

            iteration_timing["active_samples"] = len(active_indices)

            active_prompts = [
                self._build_navigator_agent_prompt(
                    question=questions[i],
                    iteration_contexts=iteration_contexts[i],
                    use_chat_template=self.navigator_agent_use_chat_template,
                )
                for i in active_indices
            ]
            phase1_started_ns = self._now_ns()
            navigator_agent_outputs = self._generate_text_batch(
                active_prompts,
                navigator_agent_config,
                llm_client=self.navigator_agent_llm_client,
                tokenizer=self.navigator_agent_tokenizer,
                use_chat_template=self.navigator_agent_use_chat_template,
                enable_thinking=self.navigator_agent_enable_thinking,
            )
            phase1_ms = self._ns_to_ms(self._now_ns() - phase1_started_ns)
            batch_phase_totals["phase1_navigator_ms"] += phase1_ms
            iteration_timing["phase1_navigator_ms"] = phase1_ms
            phase1_share = phase1_ms / len(active_indices)
            for sample_i in active_indices:
                results[sample_i]["timing"]["phase1_navigator_ms"] += phase1_share

            # Collect path-dispatch inputs from Navigator outputs.
            path_agent_generation_prompts: List[Any] = []
            path_meta: List[tuple[int, int, str, str, str, str]] = []
            path_plans_by_sample: List[List[PathAgentIterationRecord]] = [
                [] for _ in range(len(questions))
            ]
            path_records_by_sample: List[List[PathAgentIterationRecord]] = [
                [] for _ in range(len(questions))
            ]
            directions_by_sample: List[List[SearchDirection]] = [
                [] for _ in range(len(questions))
            ]
            iteration_records_by_sample: List[IterationRecord | None] = [
                None for _ in range(len(questions))
            ]

            for local_i, sample_i in enumerate(active_indices):
                navigator_agent_output = navigator_agent_outputs[local_i]
                navigator_prompt_text = self._prompt_to_text(active_prompts[local_i])
                results[sample_i]["executed_iterations"] += 1
                navigator_agent_think = self._extract_think(navigator_agent_output)
                latest_navigator_thinks[sample_i] = navigator_agent_think

                answer = self._parse_answer(navigator_agent_output)
                directions = self._parse_search_directions(navigator_agent_output)
                iteration_record: IterationRecord = {
                    "iteration": iteration_idx + 1,
                    "navigator": {
                        "prompt": navigator_prompt_text,
                        "output": navigator_agent_output,
                        "think": navigator_agent_think,
                        "answer": answer,
                        "search_directions": directions,
                    },
                    "path_agents": [],
                }
                results[sample_i]["iterations"].append(iteration_record)
                iteration_records_by_sample[sample_i] = iteration_record

                if answer:
                    results[sample_i]["final_answer"] = answer
                    completed[sample_i] = True
                    continue

                directions_by_sample[sample_i] = directions
                if not directions:
                    continue

                for path_agent_id, direction in enumerate(directions, start=1):
                    path_agent_prompt = self._build_path_agent_prompt(
                        original_question=questions[sample_i],
                        navigator_agent_think=navigator_agent_think,
                        direction=direction,
                        use_chat_template=self.path_agent_use_chat_template,
                    )
                    path_agent_generation_prompts.append(path_agent_prompt)
                    path_meta.append(
                        (
                            sample_i,
                            path_agent_id,
                            direction["direction_id"],
                            direction["direction"],
                            navigator_agent_think,
                            self._prompt_to_text(path_agent_prompt),
                        )
                    )

            iteration_timing["path_prompt_count"] = len(path_agent_generation_prompts)

            if not path_agent_generation_prompts:
                continue

            # -----------------------------
            # Phase 2: Path-agent concretization
            # -----------------------------
            phase2_started_ns = self._now_ns()
            path_agent_outputs = self._generate_text_batch(
                path_agent_generation_prompts,
                path_config,
                llm_client=self.path_agent_llm_client,
                tokenizer=self.path_agent_tokenizer,
                use_chat_template=self.path_agent_use_chat_template,
                enable_thinking=self.path_agent_enable_thinking,
            )
            phase2_ms = self._ns_to_ms(self._now_ns() - phase2_started_ns)
            batch_phase_totals["phase2_path_ms"] += phase2_ms
            iteration_timing["phase2_path_ms"] = phase2_ms

            phase2_samples = sorted({meta[0] for meta in path_meta})
            if phase2_samples:
                phase2_share = phase2_ms / len(phase2_samples)
                for sample_i in phase2_samples:
                    results[sample_i]["timing"]["phase2_path_ms"] += phase2_share

            flat_queries: List[str] = []
            query_meta: List[tuple[int, int]] = []

            for idx, path_agent_output in enumerate(path_agent_outputs):
                (
                    sample_i,
                    path_id,
                    direction_id,
                    direction_text,
                    navigator_agent_think,
                    path_prompt_text,
                ) = path_meta[idx]
                path_agent_think = self._extract_think(path_agent_output)
                path_agent_queries = self._extract_searches(path_agent_output)
                path_agent_query = path_agent_queries[0] if path_agent_queries else ""

                path_record: PathAgentIterationRecord = {
                    "prompt": path_prompt_text,
                    "path_agent_output": path_agent_output,
                    "path_id": path_id,
                    "direction_id": direction_id,
                    "direction": direction_text,
                    "navigator_agent_think": navigator_agent_think,
                    "path_agent_think": path_agent_think,
                    "search_query": path_agent_query,
                }
                path_records_by_sample[sample_i].append(path_record)

                path_plan: PathAgentIterationRecord = {
                    "prompt": path_prompt_text,
                    "path_agent_output": path_agent_output,
                    "path_id": path_id,
                    "direction_id": direction_id,
                    "direction": direction_text,
                    "navigator_agent_think": navigator_agent_think,
                    "path_agent_think": path_agent_think,
                    "search_query": path_agent_query,
                }
                path_plans_by_sample[sample_i].append(path_plan)

                if path_agent_query:
                    flat_queries.append(path_agent_query)
                    query_meta.append((sample_i, path_id))

            for sample_i in range(len(questions)):
                if (
                    iteration_records_by_sample[sample_i] is not None
                    and path_records_by_sample[sample_i]
                ):
                    iteration_record = cast(
                        IterationRecord, iteration_records_by_sample[sample_i]
                    )
                    iteration_record["path_agents"] = path_records_by_sample[sample_i]

            if not flat_queries:
                for sample_i in active_indices:
                    completed[sample_i] = True
                iteration_timing["iteration_total_ms"] = self._ns_to_ms(
                    self._now_ns() - iteration_started_ns
                )
                iteration_timings.append(iteration_timing)
                continue

            iteration_timing["retrieval_query_count"] = len(flat_queries)

            # -----------------------------
            # Phase 3: Retrieval + global documents pooling
            # -----------------------------
            phase3_retrieval_started_ns = self._now_ns()
            flat_docs = self.retriever.batch_search(flat_queries)
            phase3_retrieval_ms = self._ns_to_ms(
                self._now_ns() - phase3_retrieval_started_ns
            )
            batch_phase_totals["phase3_retrieval_ms"] += phase3_retrieval_ms
            iteration_timing["phase3_retrieval_ms"] = phase3_retrieval_ms

            phase3_samples = sorted({sample_i for sample_i, _ in query_meta})
            if phase3_samples:
                phase3_retrieval_share = phase3_retrieval_ms / len(phase3_samples)
                for sample_i in phase3_samples:
                    results[sample_i]["timing"]["phase3_retrieval_ms"] += (
                        phase3_retrieval_share
                    )

            docs_map: dict[tuple[int, int], List[RetrieverDocument]] = {}
            for idx, docs in enumerate(flat_docs):
                meta_key = query_meta[idx]
                docs_map[meta_key] = docs

            pooled_docs_by_sample: List[List[PooledDocument]] = [
                [] for _ in range(len(questions))
            ]

            # Pool documents and build iteration context (NO refine step).
            phase3_pooling_started_ns = self._now_ns()
            for sample_i in active_indices:
                sample_path_plans = path_plans_by_sample[sample_i]
                if not sample_path_plans:
                    continue
                sample_query_plans = [
                    plan for plan in sample_path_plans if plan["search_query"]
                ]
                if not sample_query_plans:
                    continue

                pooled_docs = self._pool_documents(
                    sample_query_plans, docs_map, sample_i
                )
                pooled_docs_by_sample[sample_i] = pooled_docs

                # Build iteration context and append for next Navigator call.
                ctx: IterationContext = {
                    "directions": directions_by_sample[sample_i],
                    "path_plans": sample_query_plans,
                    "pooled_docs": pooled_docs,
                }
                iteration_contexts[sample_i].append(ctx)

            phase3_pooling_ms = self._ns_to_ms(
                self._now_ns() - phase3_pooling_started_ns
            )
            batch_phase_totals["phase3_pooling_ms"] += phase3_pooling_ms
            iteration_timing["phase3_pooling_ms"] = phase3_pooling_ms

            phase3_pooling_samples = sorted(
                {
                    i
                    for i in active_indices
                    if pooled_docs_by_sample[i] or path_plans_by_sample[i]
                }
            )
            if phase3_pooling_samples:
                phase3_pooling_share = phase3_pooling_ms / len(
                    phase3_pooling_samples
                )
                for sample_i in phase3_pooling_samples:
                    results[sample_i]["timing"]["phase3_pooling_ms"] += (
                        phase3_pooling_share
                    )

            iteration_timing["iteration_total_ms"] = self._ns_to_ms(
                self._now_ns() - iteration_started_ns
            )
            iteration_timings.append(iteration_timing)

        # -----------------------------
        # Phase 4: Final answer synthesis
        # -----------------------------
        for idx, result in enumerate(results):
            if result["final_answer"]:
                continue

            final_prompt = self._build_final_answer_prompt(
                question=questions[idx],
                iteration_contexts=iteration_contexts[idx],
                use_chat_template=self.navigator_agent_use_chat_template,
            )
            final_prompt_text = self._prompt_to_text(final_prompt)
            phase5_started_ns = self._now_ns()
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
                enable_thinking=self.navigator_agent_enable_thinking,
            )
            phase5_ms = self._ns_to_ms(self._now_ns() - phase5_started_ns)
            batch_phase_totals["phase5_synthesize_ms"] += phase5_ms
            results[idx]["timing"]["phase5_synthesize_ms"] += phase5_ms
            final_output = final_outputs[0] if final_outputs else ""
            parsed = self._parse_answer(final_output)
            final_answer = ""
            if parsed:
                final_answer = parsed
            elif final_output:
                final_answer = final_output.strip()
            result["final_synthesis"] = {
                "prompt": final_prompt_text,
                "output": final_output,
                "answer": final_answer,
            }
            result["final_answer"] = final_answer

        for result in results:
            phase_sum = (
                result["timing"]["phase1_navigator_ms"]
                + result["timing"]["phase2_path_ms"]
                + result["timing"]["phase3_retrieval_ms"]
                + result["timing"]["phase3_pooling_ms"]
                + result["timing"]["phase5_synthesize_ms"]
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
