import hashlib
import re
import time
from types import SimpleNamespace
from typing import Any, Dict, List, TypedDict

from src.baseline_base import (PromptedGenerationBase,
                               build_openai_client_from_args,
                               build_retriever_client_from_args,
                               parse_stop_tokens,
                               resolve_chat_template_components)
from src.clients import (BatchSearchDocs, OpenAIClient, RetrieverClient,
                         RetrieverDocument)

THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>",
                               flags=re.DOTALL | re.IGNORECASE)
SEARCH_TAG_PATTERN = re.compile(r"<search>(.*?)</search>",
                                flags=re.DOTALL | re.IGNORECASE)
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>",
                                flags=re.DOTALL | re.IGNORECASE)
SEARCH_DIRECTIONS_PATTERN = re.compile(r"<search_directions>(.*?)</search_directions>",
                                       flags=re.DOTALL | re.IGNORECASE)
DIRECTION_TAG_PATTERN = re.compile(r"<direction(?:\s+id=[\"']?(.*?)[\"']?)?>(.*?)</direction>",
                                   flags=re.DOTALL | re.IGNORECASE)


class SearchDirection(TypedDict):
    direction_id: str
    direction: str


class PathPlan(TypedDict):
    prompt: str
    path_id: int
    direction_id: str
    direction: str
    navigator_agent_think: str
    path_agent_think: str
    search_query: str


class RefinedPathResult(TypedDict):
    path_id: int
    direction_id: str
    direction: str
    think: str
    search_query: str
    refined_information: str
    selected_doc_ids: List[str]


class PooledDocument(TypedDict):
    doc_id: str
    contents: str
    sources: List[str]
    source_directions: List[str]
    source_queries: List[str]


class ParallelO1Result(TypedDict):
    query: str
    max_iterations: int
    executed_iterations: int
    navigator_agent_pormpts: List[str]
    navigator_agent_thoughts: List[str]
    search_directions: List[List[SearchDirection]]
    path_plans: List[List[PathPlan]]
    retrieved_docs: List[BatchSearchDocs]
    pooled_docs: List[List[PooledDocument]]
    refine_prompts: List[List[str]]
    refined_paths: List[List[RefinedPathResult]]
    global_refinements: List[str]
    final_answer: str
    timing: Dict[str, Any]


class AdaptiveParallelO1(PromptedGenerationBase):
    def __init__(self,
                 retriever: RetrieverClient,
                 llm_client: OpenAIClient,
                 docs_per_query: int = 3,
                 navigator_agent_max_tokens: int = 512,
                 navigator_agent_temperature: float = 0.6,
                 navigator_agent_top_p: float = 0.9,
                 path_agent_max_tokens: int = 512,
                 path_agent_temperature: float = 0.8,
                 path_agent_top_p: float = 0.95,
                 refine_max_tokens: int = 512,
                 refine_temperature: float = 0.2,
                 refine_top_p: float = 0.9,
                 synthesize_max_tokens: int = 512,
                 synthesize_temperature: float = 0.3,
                 synthesize_top_p: float = 0.9,
                 stop_tokens: List[str] | None = None,
                 use_chat_template: bool = False,
                 tokenizer: Any = None):
        super().__init__(llm_client=llm_client,
                         generation_max_tokens=synthesize_max_tokens,
                         generation_temperature=synthesize_temperature,
                         generation_top_p=synthesize_top_p,
                         stop_tokens=stop_tokens,
                         use_chat_template=use_chat_template,
                         tokenizer=tokenizer)
        self.retriever = retriever
        self.docs_per_query = max(1, docs_per_query)
        self.navigator_agent_max_tokens = navigator_agent_max_tokens
        self.navigator_agent_temperature = navigator_agent_temperature
        self.navigator_agent_top_p = navigator_agent_top_p
        self.path_agent_max_tokens = path_agent_max_tokens
        self.path_agent_temperature = path_agent_temperature
        self.path_agent_top_p = path_agent_top_p
        self.refine_max_tokens = refine_max_tokens
        self.refine_temperature = refine_temperature
        self.refine_top_p = refine_top_p
        self.synthesize_max_tokens = synthesize_max_tokens
        self.synthesize_temperature = synthesize_temperature
        self.synthesize_top_p = synthesize_top_p
        self.latest_batch_timing: Dict[str, Any] = {}

    @classmethod
    def from_args(cls, args) -> "AdaptiveParallelO1":
        use_chat_template, tokenizer = resolve_chat_template_components(args)
        retriever_client = build_retriever_client_from_args(args)
        llm_client = build_openai_client_from_args(
            args, use_chat_template=use_chat_template)

        docs_per_query = int(getattr(args, "docs_per_query", 3))

        navigator_agent_max_tokens = int(
            getattr(args, "navigator_agent_max_tokens", 256))
        navigator_agent_temperature = float(
            getattr(args, "navigator_agent_temperature", 0.6))
        navigator_agent_top_p = float(
            getattr(args, "navigator_agent_top_p", 0.9))

        path_agent_max_tokens = int(
            getattr(args, "path_agent_max_tokens", 384))
        path_agent_temperature = float(
            getattr(args, "path_agent_temperature", 0.8))
        path_agent_top_p = float(getattr(args, "path_agent_top_p", 0.95))

        refine_max_tokens = int(getattr(args, "refine_max_tokens", 384))
        refine_temperature = float(getattr(args, "refine_temperature", 0.2))
        refine_top_p = float(getattr(args, "refine_top_p", 0.9))

        synthesize_max_tokens = int(
            getattr(args, "synthesize_max_tokens", 768))
        synthesize_temperature = float(
            getattr(args, "synthesize_temperature", 0.3))
        synthesize_top_p = float(getattr(args, "synthesize_top_p", 0.9))
        return cls(retriever=retriever_client,
                   llm_client=llm_client,
                   docs_per_query=docs_per_query,
                   navigator_agent_max_tokens=navigator_agent_max_tokens,
                   navigator_agent_temperature=navigator_agent_temperature,
                   navigator_agent_top_p=navigator_agent_top_p,
                   path_agent_max_tokens=path_agent_max_tokens,
                   path_agent_temperature=path_agent_temperature,
                   path_agent_top_p=path_agent_top_p,
                   refine_max_tokens=refine_max_tokens,
                   refine_temperature=refine_temperature,
                   refine_top_p=refine_top_p,
                   synthesize_max_tokens=synthesize_max_tokens,
                   synthesize_temperature=synthesize_temperature,
                   synthesize_top_p=synthesize_top_p,
                   stop_tokens=parse_stop_tokens(args),
                   use_chat_template=use_chat_template,
                   tokenizer=tokenizer)

    def _make_config(self, max_tokens: int, temperature: float,
                     top_p: float) -> SimpleNamespace:
        return SimpleNamespace(max_completion_length=max_tokens,
                               temperature=temperature,
                               top_p=top_p,
                               vllm_n=1)

    def _extract_searches(self, text: str) -> List[str]:
        searches: List[str] = []
        for matched in SEARCH_TAG_PATTERN.findall(text):
            q = matched.strip()
            if q and q not in searches:
                searches.append(q)
        return searches

    def _generate_text_batch(self, prompts: List[Any],
                             config: SimpleNamespace) -> List[str]:
        if not prompts:
            return []

        if not self.use_chat_template:
            prompt_texts = [self._prompt_to_text(p) for p in prompts]
            return self.llm_client.generate_text(prompt_texts,
                                                 config,
                                                 self.stop_tokens)

        return self.llm_client.generate_text(prompts,
                                             config,
                                             self.stop_tokens,
                                             tokenizer=self.tokenizer)

    def _extract_think(self, text: str) -> str:
        extracted = self._extract_last_tag(THINK_TAG_PATTERN, text)
        return extracted if extracted else text.strip()

    def _parse_search_directions(self, text: str) -> List[SearchDirection]:
        block_match = SEARCH_DIRECTIONS_PATTERN.search(text)
        if not block_match:
            return []

        block = block_match.group(1)
        directions: List[SearchDirection] = []
        for idx, match in enumerate(DIRECTION_TAG_PATTERN.finditer(block),
                                    start=1):
            direction_id = (match.group(1) or str(idx)).strip() or str(idx)
            direction_text = re.sub(r"\s+", " ", match.group(2)).strip()
            if not direction_text:
                continue
            directions.append({
                "direction_id": direction_id,
                "direction": direction_text,
            })

        return directions

    def _build_navigator_agent_prompt(self, question: str,
                                      historical_refinements: List[str]) -> Any:
        history_block = "\n\n".join([
            f"R_{idx + 1}:\n{report}"
            for idx, report in enumerate(historical_refinements)
        ]) if historical_refinements else "None"

        system_prompt = (
            "You are a navigator_agent agent in a multi-stage retrieval reasoning system to answer the user's question. "
            "You receive the original question and the historical global refinement reports R_<i>. "
            "You must first think globally inside <think>...</think>. "
            "If the available information is sufficient, output the final concise short answer inside <answer>...</answer>. "
            "If you lack of some external informatioin, you can output a dynamic set of abstract retrieval directions inside <search_directions>...</search_directions>. "
            "Each direction must be wrapped in <direction id=\"k\">...</direction>. "
            "And the system will return relative informatio inside <information>...</information>"
            "Each direction should explicitly describe the entity, relation, attribute, and the information gap that needs to be filled. "
            "You may produce as many directions as needed for this round."
        )
        user_prompt = "\n\n".join([
            self._format_external_context("Original Question", question),
            self._format_external_context("Historical Refined Information R_<i>",
                                          history_block),
            "Decide whether to answer now or propose the next retrieval directions.",
        ])
        return self._to_prompt(system_prompt, user_prompt)

    def _build_path_agent_prompt(self, original_question: str,
                                 navigator_agent_think: str,
                                 direction: SearchDirection) -> Any:
        system_prompt = (
            "You are a helpful assistant that help navigator_agent Agent to answer user's question. "
            "You will receive the original question, the navigator_agent Agent's current  thoughts, and one assigned retrieval direction by navigator_agent Agent. "
            "Your task is to convert the abstract retrieval direction into one concrete search-engine-friendly query. "
            "You must reason locally inside <think>...</think> and then output exactly one search query inside <search>...</search>. "
            "The query should be precise, operational, and directly aligned with the assigned direction."
        )
        user_prompt = "\n\n".join([
            self._format_external_context(
                "Original Question", original_question),
            self._format_external_context(
                "navigator_agent Thinking", navigator_agent_think),
            self._format_external_context(
                f"Assigned Direction {direction['direction_id']}",
                direction["direction"]),
            "Now think about above information and put your search query in <search>...</search>",
        ])
        return self._to_prompt(system_prompt, user_prompt)

    def _build_global_refine_agent_prompt(self,
                                          question: str,
                                          navigator_agent_think: str,
                                          directions: List[SearchDirection],
                                          path_plans: List[PathPlan],
                                          pooled_docs: List[PooledDocument]) -> Any:
        direction_block = "\n".join([
            f"- Direction {direction['direction_id']}: {direction['direction']}"
            for direction in directions
        ]) if directions else "None"

        query_block = "\n".join([
            f"- Direction to Search : {plan['direction_id']} | Search Query: {plan['search_query']}"
            for plan in path_plans
        ]) if path_plans else "None"

        docs_block = "\n\n".join([
            "\n".join([
                f"DocID: {doc['doc_id']}",
                f"Sources: {', '.join(doc['sources'])}",
                f"Source Directions: {' | '.join(doc['source_directions'])}",
                f"Source Queries: {' | '.join(doc['source_queries'])}",
                f"Content: {doc['contents']}",
            ])
            for doc in pooled_docs
        ]) if pooled_docs else "No documents in the pooled knowledge base."

        system_prompt = (
            "You are the Global Refine Agent.\n"
            "You are tasked with reading and analyzing document pool based on the following inputs: the original question, this round's direction set, the concrete queries used by the Path Agents, and a global document pool with provenance metadata.\n"
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
            self._format_external_context(
                "Previous navigator_agent Thinking", navigator_agent_think),
            self._format_external_context(
                "Search Directions", direction_block),
            self._format_external_context(
                "Concrete Search Queries", query_block),
            self._format_external_context(
                "Global Document Pool", docs_block),
            "Now you should analyze each web page and find  helpful information based on the original question, this round's direction set, the concrete queries, and a global document pool with provenance metadata.",
        ])
        return self._to_prompt(system_prompt, user_prompt)

    def _build_refinement_appendix(self,
                                   report: str) -> str:
        return (
            "\n<information>\n"
            f"{report}"
            "\n</information>\n"
        )

    def _build_final_answer_prompt(self, question: str,
                                   historical_refinements: List[str]) -> Any:
        history_block = "\n\n".join([
            f"R_{idx + 1}:\n{report}"
            for idx, report in enumerate(historical_refinements)
        ]) if historical_refinements else "None"
        system_prompt = (
            "You are the navigator_agent Agent. "
            "Use the historical global refinement reports to provide the best final answer now. "
            "Output only the concise final answer inside <answer>...</answer>."
        )
        user_prompt = "\n\n".join([
            self._format_external_context("Original Question", question),
            self._format_external_context(
                "Historical Global Refinements", history_block),
        ])
        return self._to_prompt(system_prompt, user_prompt)

    def _make_doc_key(self, doc: RetrieverDocument) -> str:
        doc_id = str(doc.get("id", "")).strip()
        if doc_id:
            return doc_id
        contents = str(doc.get("contents", ""))
        return hashlib.sha1(contents.encode("utf-8")).hexdigest()

    def _pool_documents(self,
                        path_plans: List[PathPlan],
                        docs_map: dict[tuple[int, int], List[RetrieverDocument]],
                        sample_i: int) -> List[PooledDocument]:
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

    def _parse_answer(self, text: str) -> str:
        return self._extract_last_tag(ANSWER_TAG_PATTERN, text)

    @staticmethod
    def _now_ns() -> int:
        return time.perf_counter_ns()

    @staticmethod
    def _ns_to_ms(duration_ns: int) -> float:
        return duration_ns / 1_000_000.0

    def _init_batch_runtime(self,
                            questions: List[str],
                            max_iterations: int) -> tuple[List[bool], List[ParallelO1Result], List[List[str]]]:
        """Initialize per-sample runtime states for batched execution."""
        completed = [False] * len(questions)

        results: List[ParallelO1Result] = [{
            "query": q,
            "max_iterations": max_iterations,
            "executed_iterations": 0,
            "navigator_agent_pormpts": [],
            "navigator_agent_thoughts": [],
            "search_directions": [],
            "path_plans": [],
            "retrieved_docs": [],
            "pooled_docs": [],
            "refine_prompts": [],
            "refined_paths": [],
            "global_refinements": [],
            "final_answer": "",
            "timing": {
                "phase1_navigator_ms": 0.0,
                "phase2_path_ms": 0.0,
                "phase3_retrieval_ms": 0.0,
                "phase3_pooling_ms": 0.0,
                "phase4_refine_ms": 0.0,
                "phase5_synthesize_ms": 0.0,
                "total_ms": 0.0,
                "executed_iterations": 0,
            },
        } for q in questions]

        # Keep historical global refinements R_<i> for each sample.
        refinement_histories: List[List[str]] = [[] for _ in questions]
        return completed, results, refinement_histories

    def run(self, question: str, max_iterations: int = 4) -> ParallelO1Result:
        return self.run_batch([question], max_iterations=max_iterations)[0]

    def run_batch(self,
                  questions: List[str],
                  max_iterations: int = 4) -> List[ParallelO1Result]:
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
            "phase4_refine_ms": 0.0,
            "phase5_synthesize_ms": 0.0,
        }
        iteration_timings: List[Dict[str, Any]] = []

        completed, results, refinement_histories = self._init_batch_runtime(
            questions=questions,
            max_iterations=max_iterations)

        # Static generation configs used in each stage.
        navigator_agent_config = self._make_config(max_tokens=self.navigator_agent_max_tokens,
                                                   temperature=self.navigator_agent_temperature,
                                                   top_p=self.navigator_agent_top_p)
        path_config = self._make_config(max_tokens=self.path_agent_max_tokens,
                                        temperature=self.path_agent_temperature,
                                        top_p=self.path_agent_top_p)
        refine_config = self._make_config(max_tokens=self.refine_max_tokens,
                                          temperature=self.refine_temperature,
                                          top_p=self.refine_top_p)

        for iteration_idx in range(max_iterations):
            iteration_started_ns = self._now_ns()
            iteration_timing: Dict[str, Any] = {
                "iteration": iteration_idx + 1,
                "active_samples": 0,
                "path_prompt_count": 0,
                "retrieval_query_count": 0,
                "refine_prompt_count": 0,
                "phase1_navigator_ms": 0.0,
                "phase2_path_ms": 0.0,
                "phase3_retrieval_ms": 0.0,
                "phase3_pooling_ms": 0.0,
                "phase4_refine_ms": 0.0,
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
                    historical_refinements=refinement_histories[i])
                for i in active_indices
            ]
            phase1_started_ns = self._now_ns()
            navigator_agent_outputs = self._generate_text_batch(active_prompts,
                                                                navigator_agent_config)
            phase1_ms = self._ns_to_ms(self._now_ns() - phase1_started_ns)
            batch_phase_totals["phase1_navigator_ms"] += phase1_ms
            iteration_timing["phase1_navigator_ms"] = phase1_ms
            phase1_share = phase1_ms / len(active_indices)
            for sample_i in active_indices:
                results[sample_i]["timing"]["phase1_navigator_ms"] += phase1_share

            # Collect path-dispatch inputs from Navigator outputs.
            path_agent_generation_prompts: List[Any] = []
            path_meta: List[tuple[int, int, str, str, str, str]] = []
            path_plans_by_sample: List[List[PathPlan]] = [
                [] for _ in range(len(questions))
            ]
            directions_by_sample: List[List[SearchDirection]] = [
                [] for _ in range(len(questions))
            ]

            for local_i, sample_i in enumerate(active_indices):
                navigator_agent_output = navigator_agent_outputs[local_i]
                if not results[sample_i]["navigator_agent_pormpts"]:
                    results[sample_i]["navigator_agent_pormpts"].append(
                        self._prompt_to_text(active_prompts[local_i]))
                results[sample_i]["navigator_agent_pormpts"].append(
                    navigator_agent_output)
                results[sample_i]["executed_iterations"] += 1
                navigator_agent_think = self._extract_think(
                    navigator_agent_output)
                results[sample_i]["navigator_agent_thoughts"].append(
                    navigator_agent_think)

                answer = self._parse_answer(navigator_agent_output)
                if answer:
                    results[sample_i]["final_answer"] = answer
                    completed[sample_i] = True
                    continue

                directions = self._parse_search_directions(
                    navigator_agent_output)
                results[sample_i]["search_directions"].append(directions)
                directions_by_sample[sample_i] = directions
                if not directions:
                    completed[sample_i] = True
                    continue

                for path_agent_id, direction in enumerate(directions, start=1):
                    path_agent_prompt = self._build_path_agent_prompt(
                        original_question=questions[sample_i],
                        navigator_agent_think=navigator_agent_think,
                        direction=direction)
                    path_agent_generation_prompts.append(path_agent_prompt)
                    path_meta.append(
                        (sample_i, path_agent_id, direction["direction_id"],
                         direction["direction"], navigator_agent_think,
                         self._prompt_to_text(path_agent_prompt)))

            iteration_timing["path_prompt_count"] = len(
                path_agent_generation_prompts)

            if not path_agent_generation_prompts:
                continue

            # -----------------------------
            # Phase 2: Path-agent concretization
            # -----------------------------
            phase2_started_ns = self._now_ns()
            path_agent_outputs = self._generate_text_batch(path_agent_generation_prompts,
                                                           path_config)
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
                sample_i, path_id, direction_id, direction_text, navigator_agent_think, path_prompt_text = path_meta[
                    idx]
                path_agent_think = self._extract_think(path_agent_output)
                path_agent_queries = self._extract_searches(path_agent_output)
                path_agent_query = path_agent_queries[0] if path_agent_queries else ""
                path_plan: PathPlan = {
                    "prompt": path_prompt_text,
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
                if path_plans_by_sample[sample_i]:
                    results[sample_i]["path_plans"].append(
                        path_plans_by_sample[sample_i])

            if not flat_queries:
                for sample_i in active_indices:
                    completed[sample_i] = True
                iteration_timing["iteration_total_ms"] = self._ns_to_ms(
                    self._now_ns() - iteration_started_ns)
                iteration_timings.append(iteration_timing)
                continue

            iteration_timing["retrieval_query_count"] = len(flat_queries)

            # -----------------------------
            # Phase 3: Retrieval + global pooling
            # -----------------------------
            phase3_retrieval_started_ns = self._now_ns()
            flat_docs = self.retriever.batch_search(flat_queries)
            phase3_retrieval_ms = self._ns_to_ms(
                self._now_ns() - phase3_retrieval_started_ns)
            batch_phase_totals["phase3_retrieval_ms"] += phase3_retrieval_ms
            iteration_timing["phase3_retrieval_ms"] = phase3_retrieval_ms

            phase3_samples = sorted({sample_i for sample_i, _ in query_meta})
            if phase3_samples:
                phase3_retrieval_share = phase3_retrieval_ms / \
                    len(phase3_samples)
                for sample_i in phase3_samples:
                    results[sample_i]["timing"]["phase3_retrieval_ms"] += phase3_retrieval_share

            docs_map: dict[tuple[int, int], List[RetrieverDocument]] = {}
            for idx, docs in enumerate(flat_docs):
                meta = query_meta[idx]
                docs_map[meta] = docs

            refine_prompts: List[Any] = []
            refine_meta: List[tuple[int, str, List[str], List[str]]] = []
            refined_paths_by_sample: List[List[RefinedPathResult]] = [
                [] for _ in range(len(questions))
            ]
            pooled_docs_by_sample: List[List[PooledDocument]] = [
                [] for _ in range(len(questions))
            ]
            retrieved_by_sample: List[BatchSearchDocs] = [
                [] for _ in range(len(questions))
            ]

            # Build one global-refine prompt for each active sample.
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

                retrieved_by_sample[sample_i] = [
                    docs_map.get((sample_i, plan["path_id"]), [])
                    for plan in sample_query_plans
                ]
                pooled_docs = self._pool_documents(sample_query_plans,
                                                   docs_map,
                                                   sample_i)
                pooled_docs_by_sample[sample_i] = pooled_docs

                refined_paths_by_sample[sample_i] = [{
                    "path_id": plan["path_id"],
                    "direction_id": plan["direction_id"],
                    "direction": plan["direction"],
                    "think": plan["path_agent_think"],
                    "search_query": plan["search_query"],
                    "refined_information": "",
                    "selected_doc_ids": [
                        doc["doc_id"] for doc in pooled_docs
                        if plan["direction_id"] in doc["sources"]
                    ],
                } for plan in sample_query_plans]

                refine_prompt = self._build_global_refine_agent_prompt(
                    question=questions[sample_i],
                    navigator_agent_think=results[sample_i]["navigator_agent_thoughts"][-1],
                    directions=directions_by_sample[sample_i],
                    path_plans=sample_query_plans,
                    pooled_docs=pooled_docs,
                )
                refine_prompts.append(refine_prompt)
                refine_meta.append((sample_i,
                                    self._prompt_to_text(refine_prompt),
                                    [doc["doc_id"] for doc in pooled_docs],
                                    [plan["direction_id"]
                                     for plan in sample_query_plans]))

            phase3_pooling_ms = self._ns_to_ms(self._now_ns() -
                                               phase3_pooling_started_ns)
            batch_phase_totals["phase3_pooling_ms"] += phase3_pooling_ms
            iteration_timing["phase3_pooling_ms"] = phase3_pooling_ms
            phase3_pooling_samples = [meta[0] for meta in refine_meta]
            if phase3_pooling_samples:
                phase3_pooling_share = phase3_pooling_ms / \
                    len(phase3_pooling_samples)
                for sample_i in phase3_pooling_samples:
                    results[sample_i]["timing"]["phase3_pooling_ms"] += phase3_pooling_share

            iteration_timing["refine_prompt_count"] = len(refine_prompts)

            # -----------------------------
            # Phase 4: Global refinement and context update
            # -----------------------------
            phase4_started_ns = self._now_ns()
            refine_agent_outputs = self._generate_text_batch(refine_prompts,
                                                             refine_config)
            phase4_ms = self._ns_to_ms(self._now_ns() - phase4_started_ns)
            batch_phase_totals["phase4_refine_ms"] += phase4_ms
            iteration_timing["phase4_refine_ms"] = phase4_ms
            phase4_samples = [meta[0] for meta in refine_meta]
            if phase4_samples:
                phase4_share = phase4_ms / len(phase4_samples)
                for sample_i in phase4_samples:
                    results[sample_i]["timing"]["phase4_refine_ms"] += phase4_share

            for idx, refine_agent_output in enumerate(refine_agent_outputs):
                sample_i, refine_prompt_text, _, _ = refine_meta[idx]
                results[sample_i]["refine_prompts"].append(
                    [refine_prompt_text])
                results[sample_i]["global_refinements"].append(
                    refine_agent_output)
                results[sample_i]["refined_paths"].append(
                    refined_paths_by_sample[sample_i])
                results[sample_i]["retrieved_docs"].append(
                    retrieved_by_sample[sample_i])
                results[sample_i]["pooled_docs"].append(
                    pooled_docs_by_sample[sample_i])
                refinement_histories[sample_i].append(refine_agent_output)
                results[sample_i]["navigator_agent_pormpts"].append(
                    self._build_refinement_appendix(refine_agent_output))

            for sample_i in active_indices:
                if not pooled_docs_by_sample[sample_i] and not completed[sample_i]:
                    completed[sample_i] = True

            iteration_timing["iteration_total_ms"] = self._ns_to_ms(
                self._now_ns() - iteration_started_ns)
            iteration_timings.append(iteration_timing)

        # -----------------------------
        # Phase 5: Final answer synthesis
        # -----------------------------
        for idx, result in enumerate(results):
            if result["final_answer"]:
                continue

            final_prompt = self._build_final_answer_prompt(
                question=questions[idx],
                historical_refinements=refinement_histories[idx])
            if not result["navigator_agent_pormpts"]:
                result["navigator_agent_pormpts"].append(
                    self._prompt_to_text(final_prompt))
            phase5_started_ns = self._now_ns()
            final_outputs = self._generate_text_batch(
                [final_prompt],
                self._make_config(max_tokens=self.synthesize_max_tokens,
                                  temperature=self.synthesize_temperature,
                                  top_p=self.synthesize_top_p))
            phase5_ms = self._ns_to_ms(self._now_ns() - phase5_started_ns)
            batch_phase_totals["phase5_synthesize_ms"] += phase5_ms
            results[idx]["timing"]["phase5_synthesize_ms"] += phase5_ms
            final_output = final_outputs[0] if final_outputs else ""
            result["navigator_agent_pormpts"].append(final_output)
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
                + result["timing"]["phase2_path_ms"]
                + result["timing"]["phase3_retrieval_ms"]
                + result["timing"]["phase3_pooling_ms"]
                + result["timing"]["phase4_refine_ms"]
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
