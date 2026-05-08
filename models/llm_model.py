"""Utilities to split sentence into ABSA clauses with Phi-3 and infer sentiment per clause."""

from __future__ import annotations

import os
from typing import Any, Dict, List

PHI3_MODEL = "microsoft/Phi-3-mini-4k-instruct:featherless-ai"


def _hf_openai_client() -> Any:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'openai'. Install by: pip install openai"
        ) from exc

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set in environment variables.")
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )


def chat_phi3(
    model: str,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.0,
) -> str:
    """Call Phi-3 through HF Router OpenAI-compatible endpoint."""
    _ = tokenizer  # Kept for API compatibility with local-model style signatures.
    client = _hf_openai_client()
    completion = client.chat.completions.create(
        model=model or PHI3_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    return (completion.choices[0].message.content or "").strip()


def split_sentence_with_terms_llm(
    sentence: str,
    model: str = PHI3_MODEL,
    tokenizer: Any = None,
    max_new_tokens: int = 300,
) -> List[Dict[str, str]]:
    """
    Split a sentence into clauses and extract term/aspect using Phi-3.
    Return list of dicts:
    [{"clause": ..., "term": ..., "sentence_original": ...}, ...]
    """
    prompt = (
        "You are an expert linguist working on Aspect-Based Sentiment Analysis (ABSA).\n"
        "Your task is to split the following review sentence into smaller clauses and identify the aspect/term discussed in each clause.\n\n"
        "==================== STRICT RULES ====================\n"
        "1. DO NOT add, remove, translate, explain, or modify ANY words, symbols, or punctuation in the original sentence.\n"
        "   - Every clause must be a continuous substring of the original sentence.\n"
        "   - The output must cover all parts of the sentence - no content should be ignored or missing.\n"
        "2. Only split the sentence where it makes sense semantically - typically around conjunctions ('and', 'but', 'while', 'although', etc.) "
        "or when the opinion changes.\n"
        "   - Do NOT split phrases that grammatically or logically belong to the same subject.\n"
        "   - If a descriptive phrase does not have a clear term in the sentence, keep it as a separate clause but leave Term blank.\n"
        "3. Keep the exact original wording and order in each clause. Do NOT reorder, paraphrase, or summarize.\n"
        "4. Each clause must express a clear opinion or evaluative meaning, either explicit or implicit.\n"
        "5. Do NOT separate adverbs from the words they modify.\n"
        "6. Keep negative or limiting words inside the same clause.\n"
        "7. Identify the TERM being discussed in each clause.\n"
        "   - TERM: the main aspect or entity being described (e.g., 'staff', 'room', 'hotel').\n"
        "   - If no clear term appears, leave it blank.\n"
        "8. Avoid creating meaningless or redundant clauses.\n"
        "9. If multiple terms appear in the same clause, separate them with commas.\n"
        "10. If a clause refers to the same entity as a previous one but does not repeat it explicitly, propagate the term from the previous clause.\n\n"
        "==================== COVERAGE REQUIREMENT ====================\n"
        "Every part of the original sentence must appear in at least one clause.\n"
        "Do NOT skip, shorten, or drop any meaningful phrase.\n\n"
        "==================== OUTPUT FORMAT ====================\n"
        "Clause: <clause text> | Term: <term1,term2,...>\n\n"
        "==================== RESPONSE INSTRUCTION ====================\n"
        "Respond ONLY with the clauses and terms exactly in the format shown above.\n"
        "Do NOT include any explanation, reasoning, or commentary.\n\n"
        f"Now process this sentence WITHOUT changing any words:\n{sentence}"
    )

    response = chat_phi3(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
    ).strip()

    result: List[Dict[str, str]] = []
    last_term = ""

    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue

        if "| Term:" in line:
            clause_text, term = line.split("| Term:", 1)
            clause_text = clause_text.replace("Clause:", "", 1).strip()
            term = term.strip()
            if term == "":
                term = last_term
            else:
                last_term = term
        else:
            clause_text = line
            term = last_term

        if clause_text:
            result.append(
                {
                    "clause": clause_text,
                    "term": term,
                    "sentence_original": sentence,
                }
            )
    return result


def clauses_to_infer_one_inputs(
    clause_items: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Convert clause+term items to infer_one input format:
    {"sentence": "<left $T$ right>", "aspect": "<term>"}.

    If a clause has multiple comma-separated terms, it emits one input per term.
    Clauses without term are skipped because infer_one requires non-empty aspect.
    """
    infer_inputs: List[Dict[str, str]] = []
    for item in clause_items:
        clause = item.get("clause", "").strip()
        term_str = item.get("term", "").strip()
        if not clause or not term_str:
            continue

        terms = [t.strip() for t in term_str.split(",") if t.strip()]
        for term in terms:
            if term not in clause:
                continue
            sentence_with_t = clause.replace(term, "$T$", 1)
            infer_inputs.append({"sentence": sentence_with_t, "aspect": term})
    return infer_inputs


def infer_sentiment_from_sentence_with_phi3_split(
    sentence: str,
    checkpoint_dir: str,
    *,
    model: str = PHI3_MODEL,
    tokenizer: Any = None,
    max_new_tokens: int = 300,
) -> List[Dict[str, Any]]:
    """
    End-to-end helper:
      1) split sentence into clause+term by Phi-3
      2) convert into infer_one-compatible train-style inputs
      3) infer sentiment for each item with PyABSA classifier
    """
    from thesis_apc_baseline.experiments.apc_inference import (
        infer_train_style_item,
        load_apc_sentiment_classifier,
    )

    clause_items = split_sentence_with_terms_llm(
        sentence=sentence,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    infer_inputs = clauses_to_infer_one_inputs(clause_items)
    classifier = load_apc_sentiment_classifier(checkpoint_dir, verbose=False)

    outputs: List[Dict[str, Any]] = []
    for item in infer_inputs:
        pred = infer_train_style_item(classifier, item, print_result=False)
        outputs.append(
            {
                "input": item,
                "prediction": pred,
            }
        )
    return outputs