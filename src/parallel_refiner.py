import re
from types import SimpleNamespace
from typing import List, TypedDict

from src.clients import (BatchSearchDocs, OpenAIClient, RetrieverClient,
                         RetrieverDocument)

SEARCH_TAG_PATTERN = re.compile(r"<search>(.*?)</search>",
                                flags=re.DOTALL | re.IGNORECASE)

ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>",
                                flags=re.DOTALL | re.IGNORECASE)


class RefinedContext(TypedDict):
    sub_query: str
    summary: str
    selected_doc_ids: List[str]


class ParallelSearchResult(TypedDict):
    query: str
    max_iterations: int
    executed_iterations: int
    prompts: List[str]
    raw_outputs: List[str]
    parallel_queries: List[List[str]]
    retrieved_docs: List[BatchSearchDocs]
    refiner_prompts: List[List[str]]
    refined_contexts: List[List[RefinedContext]]
    final_answer: str


class ParallelRetrieverRefiner:
    def __init__(self,
                 retriever: RetrieverClient,
                 llm_client: OpenAIClient,
                 parallel_query_count: int = 3,
                 docs_per_query: int = 3,
                 max_tokens: int = 256,
                 query_temperature: float = 0.7,
                 query_top_p: float = 0.9,
                 refine_max_tokens: int = 384,
                 refine_temperature: float = 0.2,
                 refine_top_p: float = 0.9,
                 synthesize_max_tokens: int = 768,
                 synthesize_temperature: float = 0.3,
                 synthesize_top_p: float = 0.9,
                 stop_tokens: List[str] | None = None):
        self.retriever = retriever
        self.llm_client = llm_client
        self.parallel_query_count = max(1, parallel_query_count)
        self.docs_per_query = max(1, docs_per_query)
        self.max_tokens = max_tokens
        self.query_temperature = query_temperature
        self.query_top_p = query_top_p
        self.refine_max_tokens = refine_max_tokens
        self.refine_temperature = refine_temperature
        self.refine_top_p = refine_top_p
        self.synthesize_max_tokens = synthesize_max_tokens
        self.synthesize_temperature = synthesize_temperature
        self.synthesize_top_p = synthesize_top_p
        self.stop_tokens = stop_tokens or []

    @classmethod
    def from_args(cls, args) -> "ParallelRetrieverRefiner":
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

        parallel_query_count = int(getattr(args, "parallel_query_count", 3))
        docs_per_query = int(getattr(args, "docs_per_query", 3))
        max_tokens = int(getattr(args, "query_max_tokens", 256))
        query_temperature = float(getattr(args, "query_temperature", 0.7))
        query_top_p = float(getattr(args, "query_top_p", 0.9))
        refine_max_tokens = int(getattr(args, "refine_max_tokens", 384))
        refine_temperature = float(getattr(args, "refine_temperature", 0.2))
        refine_top_p = float(getattr(args, "refine_top_p", 0.9))
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
                                  timeout=llm_timeout)
        return cls(retriever=retriever_client,
                   llm_client=llm_client,
                   parallel_query_count=parallel_query_count,
                   docs_per_query=docs_per_query,
                   max_tokens=max_tokens,
                   query_temperature=query_temperature,
                   query_top_p=query_top_p,
                   refine_max_tokens=refine_max_tokens,
                   refine_temperature=refine_temperature,
                   refine_top_p=refine_top_p,
                   synthesize_max_tokens=synthesize_max_tokens,
                   synthesize_temperature=synthesize_temperature,
                   synthesize_top_p=synthesize_top_p,
                   stop_tokens=stop_tokens)

    def _make_config(self, max_tokens: int, temperature: float, top_p: float) -> SimpleNamespace:
        return SimpleNamespace(max_completion_length=max_tokens,
                               temperature=temperature,
                               top_p=top_p,
                               vllm_n=1)

    def _extract_queries(self, text: str) -> List[str]:
        tagged_queries: List[str] = []
        for match in SEARCH_TAG_PATTERN.findall(text):
            candidate = match.strip()
            if candidate and candidate not in tagged_queries:
                tagged_queries.append(candidate)
        return tagged_queries

    def _build_agent_prompt(self, query: str) -> str:
        return (
            "You are an intelligent reasoning agent tasked with solving a question.\n"
            "You must systematically conduct your reasoning inside <think> and </think>.\n"
            "Based on your reasoning, decide whether you need more information or if you are ready to answer.\n"
            "If you lack information to answer the question, output your search intent by providing one or more search queries wrapped in <search> tags, for example: <search> important keywords </search>.\n"
            f"You can provide up to {self.parallel_query_count} diverse search queries at once. The diverse serach queries should "
            "The environment will return search results wrapped in <information> and </information>.\n"
            "If you determine you have accumulated enough evidence to answer the question confidently, or if no further search will help, output your final brief answer directly inside <answer> and </answer> and nothing else.\n"
            f"Original Question: {query}\n"
        )

    def _build_synthesize_prompt(self, query: str, contexts: List[RefinedContext]) -> str:
        if not contexts:
            info_block = "No additional evidence was found or retrieved."
        else:
            info_block = "\n\n".join([
                f"Sub Query: {ctx['sub_query']}\nRefined Evidence:\n{ctx['summary']}"
                for ctx in contexts
            ])

        return (
            "You are a helpful and intelligent assistant.\n"
            "Based on the following accumulated evidence, provide a concise and accurate final answer to the original question.\n"
            "Output only your actual final answer snippet concisely inside <answer> and </answer>.\n"
            f"Original Question: {query}\n\n"
            "Accumulated Evidence:\n"
            f"<information>\n{info_block}\n</information>\n"
        )

    def _build_refiner_prompt(self, query: str, sub_query: str, docs: List[RetrieverDocument]) -> str:
        docs_block = "\n\n".join([
            f"DocID: {doc.get('id', '')}\nContent: {doc.get('contents', '')}"
            for doc in docs[:self.docs_per_query]
        ])
        return (
            "You are an expert retrieval refiner.\n"
            "Given retrieved documents, extract key evidence that strongly supports the sub query or original question.\n"
            "Output concise bullet points with factual statements only.\n"
            "If there is no supporting evidence in the given documents, remove irrelevant information and summarize what you have.\n"
            f"Original Question: {query}\n"
            f"Sub Query: {sub_query}\n"
            "Retrieved Documents:\n"
            f"{docs_block}"
        )

    def refine_documents_batch(
            self, queries: List[str], all_sub_queries: List[List[str]],
            all_docs_by_query: List[BatchSearchDocs]
    ) -> tuple[List[List[RefinedContext]], List[List[str]]]:
        if not queries:
            return [], []

        flat_prompts: List[str] = []
        prompt_meta: List[tuple[int, str, List[str]]] = []
        prompts_by_sample: List[List[str]] = [[] for _ in queries]

        for sample_idx, query in enumerate(queries):
            sub_queries = all_sub_queries[sample_idx]
            docs_by_query = all_docs_by_query[sample_idx]
            for sub_idx, sub_query in enumerate(sub_queries):
                docs = docs_by_query[sub_idx] if sub_idx < len(
                    docs_by_query) else []
                trimmed_docs = docs[:self.docs_per_query]
                prompt = self._build_refiner_prompt(query=query,
                                                    sub_query=sub_query,
                                                    docs=trimmed_docs)
                flat_prompts.append(prompt)
                prompts_by_sample[sample_idx].append(prompt)
                prompt_meta.append(
                    (sample_idx, sub_query, [doc.get("id", "") for doc in trimmed_docs]))

        config = self._make_config(max_tokens=self.refine_max_tokens,
                                   temperature=self.refine_temperature,
                                   top_p=self.refine_top_p)
        summaries = self.llm_client.generate_text(
            flat_prompts, config, self.stop_tokens)
        if len(summaries) < len(flat_prompts):
            summaries.extend([""] * (len(flat_prompts) - len(summaries)))

        refined_by_sample: List[List[RefinedContext]] = [[] for _ in queries]
        for idx, (sample_idx, sub_query, selected_doc_ids) in enumerate(prompt_meta):
            refined_by_sample[sample_idx].append({
                "sub_query": sub_query,
                "summary": summaries[idx],
                "selected_doc_ids": selected_doc_ids,
            })

        return refined_by_sample, prompts_by_sample

    def run(self, question: str, max_iterations: int = 4) -> ParallelSearchResult:
        return self.run_batch([question], max_iterations=max_iterations)[0]

    def run_batch(self, questions: List[str], max_iterations: int = 4) -> List[ParallelSearchResult]:
        if not questions:
            return []

        prompts = [self._build_agent_prompt(q) for q in questions]
        completed = [False] * len(questions)

        results: List[ParallelSearchResult] = [{
            "query": q,
            "max_iterations": max_iterations,
            "executed_iterations": 0,
            "prompts": [],
            "raw_outputs": [],
            "parallel_queries": [],
            "retrieved_docs": [],
            "refiner_prompts": [],
            "refined_contexts": [],
            "final_answer": ""
        } for q in questions]

        query_config = self._make_config(max_tokens=self.max_tokens,
                                         temperature=self.query_temperature,
                                         top_p=self.query_top_p)

        for iteration in range(max_iterations):
            active_indices = [i for i, c in enumerate(completed) if not c]
            if not active_indices:
                break

            active_prompts = [prompts[i] for i in active_indices]
            raw_outputs = self.llm_client.generate_text(
                active_prompts, query_config, self.stop_tokens)

            search_indices = []
            all_queries_to_search: List[List[str]] = []

            for i, active_idx in enumerate(active_indices):
                output = raw_outputs[i]

                results[active_idx]["prompts"].append(prompts[active_idx])
                results[active_idx]["raw_outputs"].append(output)
                results[active_idx]["executed_iterations"] += 1

                prompts[active_idx] += output + "\n"

                final_answer_match = ANSWER_TAG_PATTERN.search(output)
                if final_answer_match:
                    results[active_idx]["final_answer"] = final_answer_match.group(
                        1).strip()
                    completed[active_idx] = True
                    continue

                queries = self._extract_queries(
                    output)[:self.parallel_query_count]

                if not queries:
                    completed[active_idx] = True
                    continue

                results[active_idx]["parallel_queries"].append(queries)
                search_indices.append(active_idx)
                all_queries_to_search.append(queries)

            if not search_indices:
                continue

            flat_sub_queries: List[str] = []
            spans: List[tuple[int, int]] = []
            for sub_queries in all_queries_to_search:
                start = len(flat_sub_queries)
                flat_sub_queries.extend(sub_queries)
                end = len(flat_sub_queries)
                spans.append((start, end))

            flat_docs = self.retriever.batch_search(
                flat_sub_queries) if flat_sub_queries else []
            all_docs_by_query: List[BatchSearchDocs] = []
            for start, end in spans:
                all_docs_by_query.append(flat_docs[start:end])

            refined_by_sample, refiner_prompts_by_sample = self.refine_documents_batch(
                queries=[questions[i] for i in search_indices],
                all_sub_queries=all_queries_to_search,
                all_docs_by_query=all_docs_by_query
            )

            for idx, active_idx in enumerate(search_indices):
                results[active_idx]["retrieved_docs"].append(
                    all_docs_by_query[idx])
                results[active_idx]["refiner_prompts"].append(
                    refiner_prompts_by_sample[idx])
                results[active_idx]["refined_contexts"].append(
                    refined_by_sample[idx])

                info_block = "\n\n".join([
                    f"Sub Query: {ctx['sub_query']}\nRefined Evidence:\n{ctx['summary']}"
                    for ctx in refined_by_sample[idx]
                ])
                info_prompt = f"<information>\n{info_block}\n</information>\n"
                prompts[active_idx] += info_prompt

        synthesize_prompts = []
        needs_synthesize_indices = []
        for idx in range(len(questions)):
            if not results[idx]["final_answer"]:
                needs_synthesize_indices.append(idx)
                # Combine all contexts discovered
                flat_contexts = []
                for ctx_list in results[idx]["refined_contexts"]:
                    flat_contexts.extend(ctx_list)

                synth_prompt = self._build_synthesize_prompt(
                    questions[idx], flat_contexts)
                synthesize_prompts.append(synth_prompt)
                results[idx]["prompts"].append(synth_prompt)

        if needs_synthesize_indices:
            synth_config = self._make_config(max_tokens=self.synthesize_max_tokens,
                                             temperature=self.synthesize_temperature,
                                             top_p=self.synthesize_top_p)
            synth_outputs = self.llm_client.generate_text(
                synthesize_prompts, synth_config, self.stop_tokens)

            for i, active_idx in enumerate(needs_synthesize_indices):
                out = synth_outputs[i]
                results[active_idx]["raw_outputs"].append(out)

                answer_match = ANSWER_TAG_PATTERN.search(out)
                if answer_match:
                    results[active_idx]["final_answer"] = answer_match.group(
                        1).strip()
                else:
                    match_think = re.search(
                        r"<think>(.*?)</think>", out, flags=re.DOTALL)
                    clean_output = out.replace(match_think.group(
                        0), "").strip() if match_think else out.strip()
                    results[active_idx]["final_answer"] = clean_output

        return results
