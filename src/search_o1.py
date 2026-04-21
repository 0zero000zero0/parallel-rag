import re
from types import SimpleNamespace
from typing import Any, List, TypedDict

from src.clients import (BatchSearchDocs, OpenAIClient, RetrieverClient,
                         RetrieverDocument)

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>",
                                flags=re.DOTALL | re.IGNORECASE)


class SearchStep(TypedDict):
    iteration: int
    search_query: str
    selected_doc_ids: List[str]
    refined_information: str


class SearchO1Result(TypedDict):
    query: str
    max_iterations: int
    max_search_limit: int
    executed_iterations: int
    search_count: int
    prompts: List[str]
    raw_outputs: List[str]
    search_steps: List[SearchStep]
    retrieved_docs: BatchSearchDocs
    refine_prompts: List[str]
    refine_outputs: List[str]
    final_answer: str


class SearchO1:
    def __init__(self,
                 retriever: RetrieverClient,
                 llm_client: OpenAIClient,
                 max_search_limit: int = 5,
                 docs_per_query: int = 3,
                 search_max_tokens: int = 1024,
                 search_temperature: float = 0.7,
                 search_top_p: float = 0.8,
                 refine_max_tokens: int = 512,
                 refine_temperature: float = 0.7,
                 refine_top_p: float = 0.8,
                 stop_tokens: List[str] | None = None):
        self.retriever = retriever
        self.llm_client = llm_client
        self.max_search_limit = max(1, max_search_limit)
        self.docs_per_query = max(1, docs_per_query)
        self.search_max_tokens = search_max_tokens
        self.search_temperature = search_temperature
        self.search_top_p = search_top_p
        self.refine_max_tokens = refine_max_tokens
        self.refine_temperature = refine_temperature
        self.refine_top_p = refine_top_p
        self.stop_tokens = stop_tokens or []

    @classmethod
    def from_args(cls, args) -> "SearchO1":
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

        max_search_limit = int(getattr(args, "max_search_limit", 5))
        docs_per_query = int(getattr(args, "docs_per_query", 3))

        search_max_tokens = int(getattr(args, "search_max_tokens", 1024))
        search_temperature = float(getattr(args, "search_temperature", 0.7))
        search_top_p = float(getattr(args, "search_top_p", 0.8))

        refine_max_tokens = int(getattr(args, "refine_max_tokens", 512))
        refine_temperature = float(getattr(args, "refine_temperature", 0.7))
        refine_top_p = float(getattr(args, "refine_top_p", 0.8))

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
                   max_search_limit=max_search_limit,
                   docs_per_query=docs_per_query,
                   search_max_tokens=search_max_tokens,
                   search_temperature=search_temperature,
                   search_top_p=search_top_p,
                   refine_max_tokens=refine_max_tokens,
                   refine_temperature=refine_temperature,
                   refine_top_p=refine_top_p,
                   stop_tokens=stop_tokens)

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

    def _extract_search_query(self, text: str) -> str:
        pattern = re.escape(BEGIN_SEARCH_QUERY) + r"(.*?)" + re.escape(
            END_SEARCH_QUERY)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if not matches:
            return ""
        return matches[-1].strip()

    def _build_main_agent_prompt(self, question: str) -> str:
        return (
            "You are a reasoning assistant with the ability to perform web searches to help you answer the user\'s question accurately. "
            "You have special tools: To perform a search: write <|begin_search_query|> your query here <|end_search_query|>. "
            "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format "
            "<|begin_search_result|> ...search results... <|end_search_result|>. "
            "You can repeat the search process multiple times if necessary. "
            f"The maximum number of search attempts is limited to {self.max_search_limit}. "
            "Once you have all the information you need, continue your reasoning. "
            "Remember: use <|begin_search_query|> to request a web search and end with <|end_search_query|>. "
            "When done searching, continue your reasoning.\n"
            f"Question: {question}"
        )

    def _build_refine_agent_prompt(self,
                                   prev_reasoning: str,
                                   search_query: str,
                                   docs: List[RetrieverDocument]) -> str:
        docs_block = "\n\n".join([
            f"DocID: {doc.get('id', '')}\nContent: {doc.get('contents', '')}"
            for doc in docs[:self.docs_per_query]
        ])
        if not docs_block:
            docs_block = "No retrieved documents."

        return (
            "Task Instruction: You are tasked with reading and analyzing web pages based on the following inputs: "
            "Previous Reasoning Steps, Current Search Query, and Searched Web Pages. "
            "Your objective is to extract relevant and helpful information for Current Search Query from the Searched Web Pages "
            "and seamlessly integrate this information into the Previous Reasoning Steps to continue reasoning for the original question.\n"
            "Guidelines:\n"
            "1. Analyze the Searched Web Pages:\n"
            "- Carefully review the content of each searched web page.\n"
            "- Identify factual information that is relevant to the Current Search Query and can aid in the reasoning process for the original question.\n"
            "2. Extract Relevant Information:\n"
            "- Select the information from the Searched Web Pages that directly contributes to advancing the Previous Reasoning Steps.\n"
            "- Ensure that the extracted information is accurate and relevant.\n"
            "3. Output Format:\n"
            "- If the web pages provide helpful information for current search query: Present the information beginning with 'Final Information' as shown below.\n"
            "Final Information\n[Helpful information]\n"
            "- If the web pages do not provide any helpful information for current search query: Output the following text.\n"
            "Final Information\nNo helpful information found.\n"
            "Inputs:\n"
            f"- Previous Reasoning Steps:\n{prev_reasoning}\n"
            f"- Current Search Query:\n{search_query}\n"
            f"- Searched Web Pages:\n{docs_block}\n"
            "Now you should analyze each web page and find helpful information based on the current search query "
            f"\"{search_query}\" and previous reasoning steps."
        )

    def _parse_answer(self, text: str) -> str:
        return self._extract_last_tag(ANSWER_TAG_PATTERN, text)

    def _parse_refined_information(self, text: str) -> str:
        marker = "Final Information"
        idx = text.rfind(marker)
        if idx == -1:
            cleaned = text.strip()
            return cleaned if cleaned else "No helpful information found."

        info = text[idx + len(marker):].strip()
        if not info:
            return "No helpful information found."
        return info

    def _build_search_result_appendix(self, refined_information: str) -> str:
        info = refined_information.strip() or "No helpful information found."
        return (
            f"\n{BEGIN_SEARCH_RESULT}\n"
            f"{info}\n"
            f"{END_SEARCH_RESULT}\n"
        )

    def run(self, question: str, max_iterations: int = 15) -> SearchO1Result:
        return self.run_batch([question], max_iterations=max_iterations)[0]

    def run_batch(self,
                  questions: List[str],
                  max_iterations: int = 15) -> List[SearchO1Result]:
        if not questions:
            return []

        prompts = [self._build_main_agent_prompt(q) for q in questions]
        completed = [False] * len(questions)
        search_counts = [0] * len(questions)
        executed_queries: List[set[str]] = [set() for _ in questions]

        results: List[SearchO1Result] = [{
            "query": q,
            "max_iterations": max_iterations,
            "max_search_limit": self.max_search_limit,
            "executed_iterations": 0,
            "search_count": 0,
            "prompts": [],
            "raw_outputs": [],
            "search_steps": [],
            "retrieved_docs": [],
            "refine_prompts": [],
            "refine_outputs": [],
            "final_answer": "",
        } for q in questions]

        search_config = self._make_config(max_tokens=self.search_max_tokens,
                                          temperature=self.search_temperature,
                                          top_p=self.search_top_p)
        refine_config = self._make_config(max_tokens=self.refine_max_tokens,
                                          temperature=self.refine_temperature,
                                          top_p=self.refine_top_p)

        for iteration in range(1, max_iterations + 1):
            active_indices = [i for i, c in enumerate(completed) if not c]
            if not active_indices:
                break

            active_prompts = [prompts[i] for i in active_indices]
            main_outputs = self.llm_client.generate_text(
                active_prompts, search_config, self.stop_tokens)

            queries_to_search: List[str] = []
            query_meta: List[tuple[int, str]] = []

            for local_i, sample_i in enumerate(active_indices):
                output = main_outputs[local_i]
                results[sample_i]["prompts"].append(prompts[sample_i])
                results[sample_i]["raw_outputs"].append(output)
                results[sample_i]["executed_iterations"] += 1
                prompts[sample_i] += output + "\n"

                answer = self._parse_answer(output)
                if answer:
                    results[sample_i]["final_answer"] = answer
                    completed[sample_i] = True
                    continue

                search_query = self._extract_search_query(output)
                if not search_query:
                    completed[sample_i] = True
                    continue

                if search_counts[sample_i] >= self.max_search_limit:
                    limit_message = self._build_search_result_appendix(
                        "The maximum search limit is exceeded. You are not allowed to search."
                    )
                    prompts[sample_i] += limit_message
                    results[sample_i]["raw_outputs"].append(limit_message)
                    completed[sample_i] = True
                    continue

                if search_query in executed_queries[sample_i]:
                    repeat_message = self._build_search_result_appendix(
                        "You have searched this query. Please refer to previous results."
                    )
                    prompts[sample_i] += repeat_message
                    results[sample_i]["raw_outputs"].append(repeat_message)
                    continue

                search_counts[sample_i] += 1
                executed_queries[sample_i].add(search_query)
                queries_to_search.append(search_query)
                query_meta.append((sample_i, search_query))

            if not queries_to_search:
                continue

            batch_docs = self.retriever.batch_search(queries_to_search)

            refine_prompts: List[str] = []
            refine_meta: List[tuple[int, str, List[str],
                                    List[RetrieverDocument]]] = []
            for idx, docs in enumerate(batch_docs):
                sample_i, search_query = query_meta[idx]
                selected_docs = docs[:self.docs_per_query]
                selected_doc_ids = [doc.get("id", "") for doc in selected_docs]
                prev_reasoning = prompts[sample_i]
                refine_prompt = self._build_refine_agent_prompt(
                    prev_reasoning=prev_reasoning,
                    search_query=search_query,
                    docs=selected_docs,
                )
                refine_prompts.append(refine_prompt)
                refine_meta.append((sample_i, search_query, selected_doc_ids,
                                    selected_docs))

            refine_outputs = self.llm_client.generate_text(
                refine_prompts, refine_config, self.stop_tokens)

            for idx, refine_output in enumerate(refine_outputs):
                sample_i, search_query, selected_doc_ids, selected_docs = refine_meta[idx]
                refined_info = self._parse_refined_information(refine_output)

                results[sample_i]["search_steps"].append({
                    "iteration": iteration,
                    "search_query": search_query,
                    "selected_doc_ids": selected_doc_ids,
                    "refined_information": refined_info,
                })
                results[sample_i]["retrieved_docs"].append(selected_docs)
                results[sample_i]["refine_prompts"].append(refine_prompts[idx])
                results[sample_i]["refine_outputs"].append(refine_output)
                results[sample_i]["search_count"] = search_counts[sample_i]

                prompts[sample_i] += self._build_search_result_appendix(
                    refined_info)

        for idx, result in enumerate(results):
            result["search_count"] = search_counts[idx]
            if result["final_answer"]:
                continue
            # 使用最后一次主模型输出作为最终答案；如有 <answer> 则优先提取。
            if result["raw_outputs"]:
                tail = result["raw_outputs"][-1]
                parsed = self._parse_answer(tail)
                result["final_answer"] = parsed if parsed else tail.strip()
            else:
                result["final_answer"] = ""

        return results
