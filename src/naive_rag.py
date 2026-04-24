from typing import Any, List, TypedDict

from src.baseline_base import (PromptedGenerationBase,
                               build_openai_client_from_args,
                               build_retriever_client_from_args,
                               parse_stop_tokens,
                               resolve_chat_template_components)
from src.clients import RetrieverClient, RetrieverDocument

NAIVE_RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that can solve the given question with the related documents. "
    "The realted documents are enclosed within <information> and </information> tags. "
    "Answer the given question directly without any introduction. "
    "The answer must be enclosed within <answer> and </answer> tags. "
    "In the last part of the answer, the final exact answer must be enclosed within <boxed></boxed> tag. "
    "For example, <answer> The final answer is <boxed> showt answer only</boxed> </answer>"
)


class NaiveRAGResult(TypedDict):
    query: str
    prompt: str
    retrieved_docs: List[RetrieverDocument]
    raw_output: str
    final_answer: str
    boxed_answer: str


class NaiveRAG(PromptedGenerationBase):
    def __init__(self,
                 retriever: RetrieverClient,
                 docs_per_query: int = 5,
                 **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.docs_per_query = max(1, docs_per_query)

    @classmethod
    def from_args(cls, args) -> "NaiveRAG":
        use_chat_template, tokenizer = resolve_chat_template_components(args)
        llm_client = build_openai_client_from_args(
            args, use_chat_template=use_chat_template)
        retriever = build_retriever_client_from_args(args)
        return cls(retriever=retriever,
                   docs_per_query=int(getattr(args, "docs_per_query", 5)),
                   llm_client=llm_client,
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

    def _build_information_block(self, docs: List[RetrieverDocument]) -> str:
        if not docs:
            return "<information>\nNo related documents found.\n</information>"
        doc_chunks = [
            f"DocID: {doc.get('id', '')}\nContent: {doc.get('contents', '')}"
            for doc in docs[:self.docs_per_query]
        ]
        return "<information>\n" + "\n\n".join(doc_chunks) + "\n</information>"

    def _build_prompt(self,
                      question: str,
                      docs: List[RetrieverDocument]) -> Any:
        information_block = self._build_information_block(docs)
        user_prompt = "\n\n".join([
            self._format_external_context("Question", question),
            information_block,
            "Answer the question directly using the related documents.",
        ])
        return self._to_prompt(NAIVE_RAG_SYSTEM_PROMPT, user_prompt)

    def run(self, question: str) -> NaiveRAGResult:
        return self.run_batch([question])[0]

    def run_batch(self, questions: List[str]) -> List[NaiveRAGResult]:
        retrieved_docs_batch = self.retriever.batch_search(questions)
        prompts = [
            self._build_prompt(question, docs[:self.docs_per_query])
            for question, docs in zip(questions, retrieved_docs_batch)
        ]
        outputs = self._generate_text_batch(prompts)

        results: List[NaiveRAGResult] = []
        for question, prompt, docs, output in zip(questions,
                                                  prompts,
                                                  retrieved_docs_batch,
                                                  outputs):
            selected_docs = docs[:self.docs_per_query]
            results.append({
                "query": question,
                "prompt": self._prompt_to_text(prompt),
                "retrieved_docs": selected_docs,
                "raw_output": output,
                "final_answer": self._extract_answer(output),
                "boxed_answer": self._extract_boxed(output),
            })
        return results
