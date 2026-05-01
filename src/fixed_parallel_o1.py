import re
import time
from types import SimpleNamespace
from typing import Any, TypedDict

from src.clients import (
    BatchSearchDocs,
    OpenAIClient,
    RetrieverClient,
    RetrieverDocument,
)

THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
SEARCH_TAG_PATTERN = re.compile(
    r"<search>(.*?)</search>", flags=re.DOTALL | re.IGNORECASE
)
ANSWER_TAG_PATTERN = re.compile(
    r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE
)
INFO_TAG_PATTERN = re.compile(
    r"<information>(.*?)</information>", flags=re.DOTALL | re.IGNORECASE
)


class PathPlan(TypedDict):
    prompt: str
    path_id: int
    think: str
    search_query: str


class RefinedPathResult(TypedDict):
    path_id: int
    think: str
    search_query: str
    refined_information: str
    selected_doc_ids: list[str]


class ParallelO1Result(TypedDict):
    query: str
    max_iterations: int
    executed_iterations: int
    prompts: list[Any]
    raw_outputs: list[str]
    path_plans: list[list[PathPlan]]
    retrieved_docs: list[BatchSearchDocs]
    refine_prompts: list[list[str]]
    refined_paths: list[list[RefinedPathResult]]
    global_summaries: list[str]
    final_answer: str
    timing: dict[str, Any]


class FixedParallelO1:
    def __init__(
        self,
        retriever: RetrieverClient,
        llm_client: OpenAIClient,
        parallel_path_count: int = 3,
        trigger_max_tokens: int = 256,
        trigger_temperature: float = 0.6,
        trigger_top_p: float = 0.9,
        path_max_tokens: int = 384,
        path_temperature: float = 0.8,
        path_top_p: float = 0.95,
        refine_max_tokens: int = 384,
        refine_temperature: float = 0.2,
        refine_top_p: float = 0.9,
        summarize_max_tokens: int = 512,
        summarize_temperature: float = 0.3,
        summarize_top_p: float = 0.9,
        synthesize_max_tokens: int = 768,
        synthesize_temperature: float = 0.3,
        synthesize_top_p: float = 0.9,
        stop_tokens: list[str] | None = None,
        use_chat_template: bool = False,
        tokenizer: Any = None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.parallel_path_count = max(1, parallel_path_count)
        self.trigger_max_tokens = trigger_max_tokens
        self.trigger_temperature = trigger_temperature
        self.trigger_top_p = trigger_top_p
        self.path_max_tokens = path_max_tokens
        self.path_temperature = path_temperature
        self.path_top_p = path_top_p
        self.refine_max_tokens = refine_max_tokens
        self.refine_temperature = refine_temperature
        self.refine_top_p = refine_top_p
        self.summarize_max_tokens = summarize_max_tokens
        self.summarize_temperature = summarize_temperature
        self.summarize_top_p = summarize_top_p
        self.synthesize_max_tokens = synthesize_max_tokens
        self.synthesize_temperature = synthesize_temperature
        self.synthesize_top_p = synthesize_top_p
        self.stop_tokens = stop_tokens or []
        self.latest_batch_timing: dict[str, Any] = {}

        self.use_chat_template = use_chat_template
        self.tokenizer = tokenizer
        self.llm_client.use_chat_template = use_chat_template
        if self.use_chat_template and self.tokenizer is None:
            raise ValueError("tokenizer is required when use_chat_template=True")

    @classmethod
    def from_args(cls, args) -> "FixedParallelO1":
        retriever_base_url = getattr(
            args, "retriever_base_url", "http://127.0.0.1:9000"
        )
        retriever_top_k = int(getattr(args, "retriever_top_k", 5))
        retriever_timeout = getattr(args, "retriever_timeout", None)

        openai_base_url = getattr(args, "openai_base_url", "http://127.0.0.1:8001")
        openai_api_key = getattr(args, "openai_api_key", "TEST")
        model = getattr(args, "model", None) or getattr(args, "llm_model", "Qwen3-14B")
        llm_timeout = getattr(args, "llm_timeout", None)

        use_chat_template = getattr(args, "use_chat_template", False)
        if isinstance(use_chat_template, str):
            use_chat_template = use_chat_template.lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
        else:
            use_chat_template = bool(use_chat_template)

        tokenizer = getattr(args, "tokenizer", None)
        model_path = getattr(args, "model_path", None)
        if use_chat_template and tokenizer is None and model_path:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

        parallel_path_count = int(getattr(args, "parallel_path_count", 3))

        trigger_max_tokens = int(getattr(args, "trigger_max_tokens", 256))
        trigger_temperature = float(getattr(args, "trigger_temperature", 0.6))
        trigger_top_p = float(getattr(args, "trigger_top_p", 0.9))

        path_max_tokens = int(getattr(args, "path_max_tokens", 384))
        path_temperature = float(getattr(args, "path_temperature", 0.8))
        path_top_p = float(getattr(args, "path_top_p", 0.95))

        refine_max_tokens = int(getattr(args, "refine_max_tokens", 384))
        refine_temperature = float(getattr(args, "refine_temperature", 0.2))
        refine_top_p = float(getattr(args, "refine_top_p", 0.9))

        summarize_max_tokens = int(getattr(args, "summarize_max_tokens", 512))
        summarize_temperature = float(getattr(args, "summarize_temperature", 0.3))
        summarize_top_p = float(getattr(args, "summarize_top_p", 0.9))

        synthesize_max_tokens = int(getattr(args, "synthesize_max_tokens", 768))
        synthesize_temperature = float(getattr(args, "synthesize_temperature", 0.3))
        synthesize_top_p = float(getattr(args, "synthesize_top_p", 0.9))

        stop_tokens = getattr(args, "stop_tokens", None)
        if isinstance(stop_tokens, str):
            stop_tokens = [token for token in stop_tokens.split(",") if token]

        retriever_client = RetrieverClient(
            base_url=retriever_base_url,
            top_k=retriever_top_k,
            timeout=retriever_timeout,
        )
        llm_client = OpenAIClient(
            base_url=openai_base_url,
            model=model,
            api_key=openai_api_key,
            timeout=llm_timeout,
            use_chat_template=use_chat_template,
        )
        return cls(
            retriever=retriever_client,
            llm_client=llm_client,
            parallel_path_count=parallel_path_count,
            trigger_max_tokens=trigger_max_tokens,
            trigger_temperature=trigger_temperature,
            trigger_top_p=trigger_top_p,
            path_max_tokens=path_max_tokens,
            path_temperature=path_temperature,
            path_top_p=path_top_p,
            refine_max_tokens=refine_max_tokens,
            refine_temperature=refine_temperature,
            refine_top_p=refine_top_p,
            summarize_max_tokens=summarize_max_tokens,
            summarize_temperature=summarize_temperature,
            summarize_top_p=summarize_top_p,
            synthesize_max_tokens=synthesize_max_tokens,
            synthesize_temperature=synthesize_temperature,
            synthesize_top_p=synthesize_top_p,
            stop_tokens=stop_tokens,
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

    def _extract_last_tag(self, pattern: re.Pattern[str], text: str) -> str:
        matches = pattern.findall(text)
        if not matches:
            return ""
        return matches[-1].strip()

    def _extract_searches(self, text: str) -> list[str]:
        searches: list[str] = []
        for matched in SEARCH_TAG_PATTERN.findall(text):
            q = matched.strip()
            if q and q not in searches:
                searches.append(q)
        return searches

    def _to_prompt(self, system_content: str, user_content: str) -> Any:
        if not self.use_chat_template:
            return f"{system_content}\n\n{user_content}"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _prompt_to_text(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            chunks: list[str] = []
            for msg in prompt:
                if isinstance(msg, dict):
                    role = str(msg.get("role", ""))
                    content = str(msg.get("content", ""))
                    chunks.append(f"[{role}] {content}")
            return "\n".join(chunks)
        return str(prompt)

    def _generate_text_batch(
        self, prompts: list[Any], config: SimpleNamespace
    ) -> list[str]:
        if not prompts:
            return []

        if not self.use_chat_template:
            prompt_texts = [self._prompt_to_text(p) for p in prompts]
            return self.llm_client.generate_text(prompt_texts, config, self.stop_tokens)

        return self.llm_client.generate_text(
            prompts, config, self.stop_tokens, tokenizer=self.tokenizer
        )

    def _build_main_agent_prompt(self, user_input: str) -> Any:
        system_prompt = (
            "You are a reasoning assistant with the ability to perform web searches to help you answer the user's question accurately. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find that you lack some knowledge, you can call a search engine by <search> query </search>, "
            "and it will return the refined information between <information> and </information>. "
            "You can search as many times as you want. "
            "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. "
            "For example, <answer> only short answer here </answer>."
        )
        return self._to_prompt(system_prompt, user_input)

    def _build_path_agent_prompt(
        self, state_prompt: str, original_question: str
    ) -> Any:
        main_queries = self._extract_searches(state_prompt)
        main_query = main_queries[-1] if main_queries else ""

        system_prompt = (
            "You are a path search agent on parallel mode. "
            "You are tasked to rewrite or improve the main search query into one better query. "
            "You must conduct reasoning inside <think> and </think> first and output the new search query by <search> query here </search>. "
        )
        user_prompt = (
            f"You must use the following two inputs to produce your next query:\n"
            f"- Original Question: {original_question}\n"
            f"- Main Agent Search Query: {main_query}\n\n"
            # f"Main Agent Output:\n{state_prompt}"
        )
        return self._to_prompt(system_prompt, user_prompt)

    def _build_refine_agent_prompt(
        self, question: str, think: str, query: str, docs: list[RetrieverDocument]
    ) -> Any:
        docs_block = "\n\n".join(
            [
                f"DocID: {doc.get('id', '')}\nContent: {doc.get('contents', '')}"
                for doc in docs
            ]
        )
        if not docs_block:
            docs_block = "No retrieved documents."

        system_prompt = (
            "You are tasked with reading and analyzing web pages based on the following inputs: Previous Reasoning Steps, Current Search Query, and Searched Web Pages. "
            "Your objective is to extract relevant and helpful information for Current Search Query from the Searched Web Pages and seamlessly integrate this information into the Previous Reasoning Steps to continue reasoning for the original question.\n"
            "Guidelines:\n"
            "1. Analyze the Searched Web Pages:\n"
            "- Carefully review the content of each searched web page.\n"
            "- Identify factual information that is relevant to the Current Search Query and can aid in the reasoning process for the original question.\n"
            "2. Extract Relevant Information:\n"
            "- Select the information from the Searched Web Pages that directly contributes to advancing the Previous Reasoning Steps.\n"
            "- Ensure that the extracted information is accurate and relevant.\n"
            "3. Output Format:\n"
            "- If the web pages provide helpful information for current search query and original question: Present the information between <information> and </information>. For example, <information> Helpful information </information>\n"
            "- If the web pages do not provide any helpful information for current search query and original question: Output <information>No helpful information found.</information>"
        )
        user_prompt = (
            f"Inputs:\n"
            f"Original Question: {question}\n"
            f"Previous Reasoning Steps: {think}\n"
            f"Current Search Query: {query}\n"
            f"Searched Web Pages: {docs_block}\n\n"
            "Now analyze each web page and find helpful information."
        )
        return self._to_prompt(system_prompt, user_prompt)

    def _build_parallel_appendix(self, refined_paths: list[RefinedPathResult]) -> str:
        if not refined_paths:
            return "\n<information>\nNo helpful information found.\n</information>\n"

        blocks: list[str] = []
        for path in refined_paths:
            blocks.append(
                "\n".join(
                    [
                        f"Path: p_{path['path_id']}",
                        f"<think>{path['think']}</think>",
                        f"<search>{path['search_query']}</search>",
                        f"<information>{path['refined_information']}</information>",
                    ]
                )
            )
        merged = "\n\n".join(blocks)
        return f"\n<information>\n{merged}\n</information>\n"

    def _parse_answer(self, text: str) -> str:
        return self._extract_last_tag(ANSWER_TAG_PATTERN, text)

    def _parse_info(self, text: str) -> str:
        extracted = self._extract_last_tag(INFO_TAG_PATTERN, text)
        return extracted if extracted else text.strip()

    @staticmethod
    def _now_ns() -> int:
        return time.perf_counter_ns()

    @staticmethod
    def _ns_to_ms(duration_ns: int) -> float:
        return duration_ns / 1_000_000.0

    def run(self, question: str, max_iterations: int = 4) -> ParallelO1Result:
        return self.run_batch([question], max_iterations=max_iterations)[0]

    def run_batch(
        self, questions: list[str], max_iterations: int = 4
    ) -> list[ParallelO1Result]:
        if not questions:
            return []

        run_batch_started_ns = self._now_ns()
        batch_phase_totals: dict[str, float] = {
            "phase1_trigger_ms": 0.0,
            "phase2_path_ms": 0.0,
            "phase3_retrieval_ms": 0.0,
            "phase4_refine_ms": 0.0,
            "phase5_finalize_ms": 0.0,
        }
        iteration_timings: list[dict[str, Any]] = []

        dialogue_states = [f"User's Question: {q}" for q in questions]
        completed = [False] * len(questions)

        results: list[ParallelO1Result] = [
            {
                "query": q,
                "max_iterations": max_iterations,
                "executed_iterations": 0,
                "prompts": [],
                "raw_outputs": [],
                "path_plans": [],
                "retrieved_docs": [],
                "refine_prompts": [],
                "refined_paths": [],
                "global_summaries": [],
                "final_answer": "",
                "timing": {
                    "phase1_trigger_ms": 0.0,
                    "phase2_path_ms": 0.0,
                    "phase3_retrieval_ms": 0.0,
                    "phase4_refine_ms": 0.0,
                    "phase5_finalize_ms": 0.0,
                    "total_ms": 0.0,
                    "executed_iterations": 0,
                },
            }
            for q in questions
        ]

        trigger_config = self._make_config(
            max_tokens=self.trigger_max_tokens,
            temperature=self.trigger_temperature,
            top_p=self.trigger_top_p,
        )
        path_config = self._make_config(
            max_tokens=self.path_max_tokens,
            temperature=self.path_temperature,
            top_p=self.path_top_p,
        )
        refine_config = self._make_config(
            max_tokens=self.refine_max_tokens,
            temperature=self.refine_temperature,
            top_p=self.refine_top_p,
        )

        for iteration_idx in range(max_iterations):
            iteration_started_ns = self._now_ns()
            iteration_timing: dict[str, Any] = {
                "iteration": iteration_idx + 1,
                "active_samples": 0,
                "path_prompt_count": 0,
                "retrieval_query_count": 0,
                "refine_prompt_count": 0,
                "phase1_trigger_ms": 0.0,
                "phase2_path_ms": 0.0,
                "phase3_retrieval_ms": 0.0,
                "phase4_refine_ms": 0.0,
                "iteration_total_ms": 0.0,
            }
            active_indices = [i for i, c in enumerate(completed) if not c]
            if not active_indices:
                break

            iteration_timing["active_samples"] = len(active_indices)

            active_prompts = [
                self._build_main_agent_prompt(dialogue_states[i])
                for i in active_indices
            ]
            phase1_started_ns = self._now_ns()
            trigger_outputs = self._generate_text_batch(active_prompts, trigger_config)
            phase1_ms = self._ns_to_ms(self._now_ns() - phase1_started_ns)
            batch_phase_totals["phase1_trigger_ms"] += phase1_ms
            iteration_timing["phase1_trigger_ms"] = phase1_ms
            phase1_share = phase1_ms / len(active_indices)
            for sample_i in active_indices:
                results[sample_i]["timing"]["phase1_trigger_ms"] += phase1_share

            path_generation_prompts: list[Any] = []
            path_meta: list[tuple[int, int, str]] = []

            for local_i, sample_i in enumerate(active_indices):
                trigger_output = trigger_outputs[local_i]
                results[sample_i]["prompts"].append(active_prompts[local_i])
                results[sample_i]["raw_outputs"].append(trigger_output)
                results[sample_i]["executed_iterations"] += 1
                dialogue_states[sample_i] += trigger_output + "\n"

                answer = self._parse_answer(trigger_output)
                if answer:
                    results[sample_i]["final_answer"] = answer
                    completed[sample_i] = True
                    continue

                for path_id in range(1, self.parallel_path_count + 1):
                    path_state = (
                        f"Original Question: {questions[sample_i]}\n"
                        f"Main Agent Output:\n{trigger_output}\n"
                    )
                    path_prompt = self._build_path_agent_prompt(
                        path_state, results[sample_i]["query"]
                    )
                    path_generation_prompts.append(path_prompt)
                    path_meta.append(
                        (sample_i, path_id, self._prompt_to_text(path_prompt))
                    )

            iteration_timing["path_prompt_count"] = len(path_generation_prompts)

            if not path_generation_prompts:
                iteration_timing["iteration_total_ms"] = self._ns_to_ms(
                    self._now_ns() - iteration_started_ns
                )
                iteration_timings.append(iteration_timing)
                continue

            phase2_started_ns = self._now_ns()
            path_outputs = self._generate_text_batch(
                path_generation_prompts, path_config
            )
            phase2_ms = self._ns_to_ms(self._now_ns() - phase2_started_ns)
            batch_phase_totals["phase2_path_ms"] += phase2_ms
            iteration_timing["phase2_path_ms"] = phase2_ms

            phase2_samples = sorted({meta[0] for meta in path_meta})
            if phase2_samples:
                phase2_share = phase2_ms / len(phase2_samples)
                for sample_i in phase2_samples:
                    results[sample_i]["timing"]["phase2_path_ms"] += phase2_share

            path_plans_by_sample: list[list[PathPlan]] = [
                [] for _ in range(len(questions))
            ]

            flat_queries: list[str] = []
            query_meta: list[tuple[int, int]] = []

            for idx, output in enumerate(path_outputs):
                sample_i, path_id, path_prompt_text = path_meta[idx]
                think = self._extract_last_tag(THINK_TAG_PATTERN, output)
                if think == "":
                    think = output
                queries = self._extract_searches(output)
                query = queries[0] if queries else ""
                path_plan: PathPlan = {
                    "prompt": path_prompt_text,
                    "path_id": path_id,
                    "think": think,
                    "search_query": query,
                }
                path_plans_by_sample[sample_i].append(path_plan)

                if query:
                    flat_queries.append(query)
                    query_meta.append((sample_i, path_id))

            iteration_timing["retrieval_query_count"] = len(flat_queries)

            for sample_i in range(len(questions)):
                if path_plans_by_sample[sample_i]:
                    results[sample_i]["path_plans"].append(
                        path_plans_by_sample[sample_i]
                    )

            if not flat_queries:
                for sample_i in active_indices:
                    completed[sample_i] = True
                iteration_timing["iteration_total_ms"] = self._ns_to_ms(
                    self._now_ns() - iteration_started_ns
                )
                iteration_timings.append(iteration_timing)
                continue

            phase3_started_ns = self._now_ns()
            flat_docs = self.retriever.batch_search(flat_queries)
            phase3_ms = self._ns_to_ms(self._now_ns() - phase3_started_ns)
            batch_phase_totals["phase3_retrieval_ms"] += phase3_ms
            iteration_timing["phase3_retrieval_ms"] = phase3_ms

            phase3_samples = sorted({sample_i for sample_i, _ in query_meta})
            if phase3_samples:
                phase3_share = phase3_ms / len(phase3_samples)
                for sample_i in phase3_samples:
                    results[sample_i]["timing"]["phase3_retrieval_ms"] += phase3_share

            docs_map: dict[tuple[int, int], list[RetrieverDocument]] = {}
            for idx, docs in enumerate(flat_docs):
                meta = query_meta[idx]
                docs_map[meta] = docs

            refine_prompts: list[Any] = []
            refine_meta: list[tuple[int, int, str, str, list[str]]] = []

            for sample_i in active_indices:
                sample_path_plans = path_plans_by_sample[sample_i]
                if not sample_path_plans:
                    continue
                for plan in sample_path_plans:
                    key = (sample_i, plan["path_id"])
                    docs = docs_map.get(key, [])
                    selected_doc_ids = [doc.get("id", "") for doc in docs]
                    refine_prompt = self._build_refine_agent_prompt(
                        question=questions[sample_i],
                        think=plan["think"],
                        query=plan["search_query"],
                        docs=docs,
                    )
                    refine_prompts.append(refine_prompt)
                    refine_meta.append(
                        (
                            sample_i,
                            plan["path_id"],
                            plan["think"],
                            plan["search_query"],
                            selected_doc_ids,
                        )
                    )

            iteration_timing["refine_prompt_count"] = len(refine_prompts)

            phase4_started_ns = self._now_ns()
            refine_outputs = self._generate_text_batch(refine_prompts, refine_config)
            phase4_ms = self._ns_to_ms(self._now_ns() - phase4_started_ns)
            batch_phase_totals["phase4_refine_ms"] += phase4_ms
            iteration_timing["phase4_refine_ms"] = phase4_ms

            phase4_samples = [meta[0] for meta in refine_meta]
            if phase4_samples:
                phase4_share = phase4_ms / len(phase4_samples)
                for sample_i in phase4_samples:
                    results[sample_i]["timing"]["phase4_refine_ms"] += phase4_share

            refined_paths_by_sample: list[list[RefinedPathResult]] = [
                [] for _ in range(len(questions))
            ]
            retrieved_by_sample: list[BatchSearchDocs] = [
                [] for _ in range(len(questions))
            ]
            refine_prompts_by_sample: list[list[Any]] = [
                [] for _ in range(len(questions))
            ]

            for idx, refine_output in enumerate(refine_outputs):
                sample_i, path_id, think, search_query, selected_doc_ids = refine_meta[
                    idx
                ]
                info = self._parse_info(refine_output)

                refined_paths_by_sample[sample_i].append(
                    {
                        "path_id": path_id,
                        "think": think,
                        "search_query": search_query,
                        "refined_information": info,
                        "selected_doc_ids": selected_doc_ids,
                    }
                )
                refine_prompts_by_sample[sample_i].append(refine_prompts[idx])
                retrieved_by_sample[sample_i].append(
                    docs_map.get((sample_i, path_id), [])
                )

            for sample_i in active_indices:
                refined_paths = refined_paths_by_sample[sample_i]
                if not refined_paths:
                    continue

                results[sample_i]["retrieved_docs"].append(
                    retrieved_by_sample[sample_i]
                )
                results[sample_i]["refine_prompts"].append(
                    [
                        self._prompt_to_text(p)
                        for p in refine_prompts_by_sample[sample_i]
                    ]
                )
                results[sample_i]["refined_paths"].append(refined_paths)

                dialogue_states[sample_i] += self._build_parallel_appendix(
                    refined_paths
                )

            iteration_timing["iteration_total_ms"] = self._ns_to_ms(
                self._now_ns() - iteration_started_ns
            )
            iteration_timings.append(iteration_timing)

        phase5_started_ns = self._now_ns()
        for idx, result in enumerate(results):
            if result["final_answer"]:
                continue
            if result["raw_outputs"]:
                tail = result["raw_outputs"][-1]
                parsed = self._parse_answer(tail)
                result["final_answer"] = parsed if parsed else tail.strip()
            else:
                result["final_answer"] = ""

        phase5_ms = self._ns_to_ms(self._now_ns() - phase5_started_ns)
        batch_phase_totals["phase5_finalize_ms"] += phase5_ms
        if results:
            phase5_share = phase5_ms / len(results)
            for result in results:
                result["timing"]["phase5_finalize_ms"] += phase5_share
                result["timing"]["total_ms"] = (
                    result["timing"]["phase1_trigger_ms"]
                    + result["timing"]["phase2_path_ms"]
                    + result["timing"]["phase3_retrieval_ms"]
                    + result["timing"]["phase4_refine_ms"]
                    + result["timing"]["phase5_finalize_ms"]
                )
                result["timing"]["executed_iterations"] = result["executed_iterations"]

        self.latest_batch_timing = {
            "num_questions": len(questions),
            "max_iterations": max_iterations,
            "total_ms": self._ns_to_ms(self._now_ns() - run_batch_started_ns),
            "phase_totals_ms": batch_phase_totals,
            "iteration_timings": iteration_timings,
        }

        return results
