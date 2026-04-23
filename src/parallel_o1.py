import hashlib
import re
from types import SimpleNamespace
from typing import Any, List, TypedDict

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


class ParallelO1:
    def __init__(self,
                 retriever: RetrieverClient,
                 llm_client: OpenAIClient,
                 docs_per_query: int = 3,
                 navigator_agent_max_tokens: int = 256,
                 navigator_agent_temperature: float = 0.6,
                 navigator_agent_top_p: float = 0.9,
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
                 stop_tokens: List[str] | None = None,
                 use_chat_template: bool = False,
                 tokenizer: Any = None):
        self.retriever = retriever
        self.llm_client = llm_client
        self.docs_per_query = max(1, docs_per_query)
        self.navigator_agent_max_tokens = navigator_agent_max_tokens
        self.navigator_agent_temperature = navigator_agent_temperature
        self.navigator_agent_top_p = navigator_agent_top_p
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

        self.use_chat_template = use_chat_template
        self.tokenizer = tokenizer
        self.llm_client.use_chat_template = use_chat_template
        if self.use_chat_template and self.tokenizer is None:
            raise ValueError(
                "tokenizer is required when use_chat_template=True")

    @classmethod
    def from_args(cls, args) -> "ParallelO1":
        retriever_base_url = getattr(
            args, "retriever_base_url", "http://127.0.0.1:9000")
        retriever_top_k = int(getattr(args, "retriever_top_k", 5))
        retriever_timeout = getattr(args, "retriever_timeout", None)

        openai_base_url = getattr(
            args, "openai_base_url", "http://127.0.0.1:8001")
        openai_api_key = getattr(args, "openai_api_key", "TEST")
        model = getattr(args, "model", None) or getattr(
            args, "llm_model", "Qwen3-14B")
        llm_timeout = getattr(args, "llm_timeout", None)

        use_chat_template = getattr(args, "use_chat_template", False)
        if isinstance(use_chat_template, str):
            use_chat_template = use_chat_template.lower() in {
                "1", "true", "yes", "y", "on"
            }
        else:
            use_chat_template = bool(use_chat_template)

        tokenizer = getattr(args, "tokenizer", None)
        model_path = getattr(args, "model_path", None)
        if use_chat_template and tokenizer is None and model_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                      trust_remote_code=True)

        docs_per_query = int(getattr(args, "docs_per_query", 3))

        navigator_agent_max_tokens = int(
            getattr(args, "navigator_agent_max_tokens", 256))
        navigator_agent_temperature = float(
            getattr(args, "navigator_agent_temperature", 0.6))
        navigator_agent_top_p = float(
            getattr(args, "navigator_agent_top_p", 0.9))

        path_max_tokens = int(getattr(args, "path_max_tokens", 384))
        path_temperature = float(getattr(args, "path_temperature", 0.8))
        path_top_p = float(getattr(args, "path_top_p", 0.95))

        refine_max_tokens = int(getattr(args, "refine_max_tokens", 384))
        refine_temperature = float(getattr(args, "refine_temperature", 0.2))
        refine_top_p = float(getattr(args, "refine_top_p", 0.9))

        summarize_max_tokens = int(getattr(args, "summarize_max_tokens", 512))
        summarize_temperature = float(
            getattr(args, "summarize_temperature", 0.3))
        summarize_top_p = float(getattr(args, "summarize_top_p", 0.9))

        synthesize_max_tokens = int(
            getattr(args, "synthesize_max_tokens", 768))
        synthesize_temperature = float(
            getattr(args, "synthesize_temperature", 0.3))
        synthesize_top_p = float(getattr(args, "synthesize_top_p", 0.9))

        stop_tokens = getattr(args, "stop_tokens", None)
        if isinstance(stop_tokens, str):
            stop_tokens = [token for token in stop_tokens.split(",") if token]

        retriever_client = RetrieverClient(base_url=retriever_base_url,
                                           top_k=retriever_top_k,
                                           timeout=retriever_timeout)
        llm_client = OpenAIClient(base_url=openai_base_url,
                                  model=model,
                                  api_key=openai_api_key,
                                  timeout=llm_timeout,
                                  use_chat_template=use_chat_template)
        return cls(retriever=retriever_client,
                   llm_client=llm_client,
                   docs_per_query=docs_per_query,
                   navigator_agent_max_tokens=navigator_agent_max_tokens,
                   navigator_agent_temperature=navigator_agent_temperature,
                   navigator_agent_top_p=navigator_agent_top_p,
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
                   tokenizer=tokenizer)

    def _make_config(self, max_tokens: int, temperature: float,
                     top_p: float) -> SimpleNamespace:
        return SimpleNamespace(max_completion_length=max_tokens,
                               temperature=temperature,
                               top_p=top_p,
                               vllm_n=1)

    def _extract_last_tag(self, pattern: re.Pattern[str], text: str) -> str:
        matches = pattern.findall(text)
        if not matches:
            return ""
        return matches[-1].strip()

    def _extract_searches(self, text: str) -> List[str]:
        searches: List[str] = []
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
            chunks: List[str] = []
            for msg in prompt:
                if isinstance(msg, dict):
                    role = str(msg.get("role", ""))
                    content = str(msg.get("content", ""))
                    chunks.append(f"[{role}] {content}")
            return "\n".join(chunks)
        return str(prompt)

    def _format_external_context(self, label: str, content: str) -> str:
        cleaned = (content or "").strip()
        if not cleaned:
            cleaned = "None"
        if "\n" in cleaned:
            return f"{label}:\n```\n{cleaned}\n```"
        return f"{label}: `{cleaned}`"

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

        completed, results, refinement_histories = self._init_batch_runtime(
            questions=questions,
            max_iterations=max_iterations)

        # Static generation configs used in each stage.
        navigator_agent_config = self._make_config(max_tokens=self.navigator_agent_max_tokens,
                                                   temperature=self.navigator_agent_temperature,
                                                   top_p=self.navigator_agent_top_p)
        path_config = self._make_config(max_tokens=self.path_max_tokens,
                                        temperature=self.path_temperature,
                                        top_p=self.path_top_p)
        refine_config = self._make_config(max_tokens=self.refine_max_tokens,
                                          temperature=self.refine_temperature,
                                          top_p=self.refine_top_p)

        for iteration_idx in range(max_iterations):
            # -----------------------------
            # Phase 1: Navigator planning
            # -----------------------------
            active_indices = [i for i, c in enumerate(completed) if not c]
            if not active_indices:
                break

            active_prompts = [
                self._build_navigator_agent_prompt(
                    question=questions[i],
                    historical_refinements=refinement_histories[i])
                for i in active_indices
            ]
            navigator_agent_outputs = self._generate_text_batch(active_prompts,
                                                                navigator_agent_config)

            # Collect path-dispatch inputs from Navigator outputs.
            path_generation_prompts: List[Any] = []
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

                for path_id, direction in enumerate(directions, start=1):
                    path_prompt = self._build_path_agent_prompt(
                        original_question=questions[sample_i],
                        navigator_agent_think=navigator_agent_think,
                        direction=direction)
                    path_generation_prompts.append(path_prompt)
                    path_meta.append(
                        (sample_i, path_id, direction["direction_id"],
                         direction["direction"], navigator_agent_think,
                         self._prompt_to_text(path_prompt)))

            if not path_generation_prompts:
                continue

            # -----------------------------
            # Phase 2: Path-agent concretization
            # -----------------------------
            path_outputs = self._generate_text_batch(path_generation_prompts,
                                                     path_config)

            flat_queries: List[str] = []
            query_meta: List[tuple[int, int]] = []

            for idx, path_output in enumerate(path_outputs):
                sample_i, path_id, direction_id, direction_text, navigator_agent_think, path_prompt_text = path_meta[
                    idx]
                path_agent_think = self._extract_think(path_output)
                path_agent_queries = self._extract_searches(path_output)
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
                continue

            # -----------------------------
            # Phase 3: Retrieval + global pooling
            # -----------------------------
            flat_docs = self.retriever.batch_search(flat_queries)

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

            # -----------------------------
            # Phase 4: Global refinement and context update
            # -----------------------------
            refine_agent_outputs = self._generate_text_batch(refine_prompts,
                                                             refine_config)

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
            final_outputs = self._generate_text_batch(
                [final_prompt],
                self._make_config(max_tokens=self.synthesize_max_tokens,
                                  temperature=self.synthesize_temperature,
                                  top_p=self.synthesize_top_p))
            final_output = final_outputs[0] if final_outputs else ""
            result["navigator_agent_pormpts"].append(final_output)
            parsed = self._parse_answer(final_output)
            if parsed:
                result["final_answer"] = parsed
            elif final_output:
                result["final_answer"] = final_output.strip()
            else:
                result["final_answer"] = ""

        return results
