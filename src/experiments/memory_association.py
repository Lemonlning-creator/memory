"""
Experiments 3a, 3b, 3c: Memory Association Validation

3a - MEMORY PROBING: Use the user profile to answer REALTALK QA questions.
     Compare profile-guided retrieval vs random retrieval vs no retrieval.
     Shows profile tags help connect scattered evidence to answer questions.

3b - PROFILE-EVIDENCE CHAIN: For multi-evidence QA pairs, show that the
     profile can explain WHY disparate evidence items are connected.

3c - ABLATION: Profile-tagged association vs random association. Measure
     whether profile-guided memory selection produces better answers.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..llm_client import LLMClient
from ..utils import load_json, save_json
from ..prompts.prompt_loader import PROFILE_EXTRACTION_SYSTEM_PROMPT
from .persona_simulation import detect_speakers, flatten_messages, session_keys


def _parse_json(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r'"\s*\n\s*"', '",\n"', raw)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            return {"error": str(e), "raw": raw[:300]}


# ---------------------------------------------------------------------------
# Build evidence lookup: map dia_id -> message text
# ---------------------------------------------------------------------------

def build_evidence_map(chat: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map dia_id -> {speaker, text, session}."""
    dia_map = {}
    for s_key in session_keys(chat):
        for msg in chat.get(s_key, []):
            dia_id = msg.get("dia_id", "")
            if dia_id:
                dia_map[dia_id] = {
                    "speaker": msg.get("speaker", ""),
                    "text": str(msg.get("clean_text", "")).strip(),
                    "session": s_key,
                }
    return dia_map


def get_all_message_ids(chat: Dict[str, Any]) -> List[str]:
    """Get all dia_ids in the conversation."""
    ids = []
    for s_key in session_keys(chat):
        for msg in chat.get(s_key, []):
            dia_id = msg.get("dia_id", "")
            if dia_id:
                ids.append(dia_id)
    return ids


def condense_profile_for_retrieval(profile: Dict[str, Any]) -> str:
    """Flatten profile into a text string for retrieval guidance."""
    lines = []
    for section, tags in profile.items():
        if isinstance(tags, dict):
            for tag_name, tag_val in tags.items():
                if isinstance(tag_val, dict):
                    val = tag_val.get("value", "")
                    if val and val.strip():
                        lines.append(f"- [{section}/{tag_name}] {val}")
                elif isinstance(tag_val, str) and tag_val.strip():
                    lines.append(f"- [{section}/{tag_name}] {tag_val}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment 3a: Memory Probing
# ---------------------------------------------------------------------------

QA_ANSWER_SYSTEM_PROMPT = """You are answering a question about a person based on conversation evidence.

You will be given:
1. A question about the person
2. A set of evidence messages (selected from the conversation)

Answer the question based ONLY on the evidence provided. If the evidence is insufficient, say "Insufficient evidence."

Output ONLY your answer, no explanation."""

QA_ANSWER_PROFILE_SYSTEM_PROMPT = """You are answering a question about a person using their user profile to guide evidence retrieval.

You will be given:
1. A question about the person
2. The person's user profile (personality traits, preferences, history)
3. A set of candidate evidence messages

FIRST: Use the profile to identify which profile tags are relevant to the question.
SECOND: Select the evidence messages that are most relevant based on those profile tags.
THIRD: Answer the question using the selected evidence.

Output ONLY your answer, no explanation."""

QA_EVAL_SYSTEM_PROMPT = """You are evaluating whether an answer to a question about a person is correct.

You will be given:
1. The question
2. The ground truth answer
3. The predicted answer

Score the predicted answer:
- 1: Correct (contains the key information from the ground truth)
- 0.5: Partially correct (some correct information but missing key details)
- 0: Incorrect (wrong or insufficient information)

Output ONLY valid JSON: {"score": <0|0.5|1>, "reasoning": "brief explanation"}"""


def run_memory_probing(
    llm: LLMClient,
    chat: Dict[str, Any],
    profile: Dict[str, Any],
    dia_map: Dict[str, Dict[str, Any]],
    all_ids: List[str],
    max_qa: int = 10,
) -> List[Dict[str, Any]]:
    """Run memory probing on one conversation's QA pairs."""
    qa_pairs = chat.get("qa", [])
    # Focus on category 1 (profile) and 3 (preference) questions
    profile_qa = [q for q in qa_pairs if q.get("category") in (1, 3)]
    random.seed(42)
    random.shuffle(profile_qa)
    selected = profile_qa[:max_qa]

    results = []
    for qa in selected:
        question = qa["question"]
        gt_answer = qa["answer"]
        gt_evidence = qa.get("evidence", [])

        # Method 1: Profile-guided (give profile + all messages from relevant sessions)
        profile_text = condense_profile_for_retrieval(profile)
        # Select a candidate pool: the GT evidence + some distractors
        gt_set = set(gt_evidence)
        distractors = [eid for eid in all_ids if eid not in gt_set]
        random.shuffle(distractors)
        candidate_ids = gt_evidence + distractors[:min(10, len(distractors))]
        candidate_msgs = "\n".join(
            f"[{eid}] {dia_map[eid]['speaker']}: {dia_map[eid]['text']}"
            for eid in candidate_ids if eid in dia_map
        )

        # Profile-guided answer
        prof_prompt = (
            f"USER PROFILE:\n{profile_text}\n\n"
            f"CANDIDATE EVIDENCE:\n{candidate_msgs}\n\n"
            f"QUESTION: {question}"
        )
        profile_answer = llm.chat(
            QA_ANSWER_PROFILE_SYSTEM_PROMPT, prof_prompt,
            temperature=0.3, max_tokens=200,
        ).strip()

        # Random-guided answer (same evidence pool, no profile)
        random_prompt = (
            f"EVIDENCE:\n{candidate_msgs}\n\n"
            f"QUESTION: {question}"
        )
        random_answer = llm.chat(
            QA_ANSWER_SYSTEM_PROMPT, random_prompt,
            temperature=0.3, max_tokens=200,
        ).strip()

        # Evaluate both
        prof_eval = _parse_json(llm.chat(
            QA_EVAL_SYSTEM_PROMPT,
            f"Question: {question}\nGround truth: {gt_answer}\nPredicted: {profile_answer}",
            temperature=0.1, max_tokens=200,
        ))
        rand_eval = _parse_json(llm.chat(
            QA_EVAL_SYSTEM_PROMPT,
            f"Question: {question}\nGround truth: {gt_answer}\nPredicted: {random_answer}",
            temperature=0.1, max_tokens=200,
        ))

        result = {
            "question": question,
            "ground_truth": gt_answer,
            "category": qa.get("category"),
            "num_gt_evidence": len(gt_evidence),
            "profile_answer": profile_answer,
            "profile_score": prof_eval.get("score", 0),
            "random_answer": random_answer,
            "random_score": rand_eval.get("score", 0),
        }
        results.append(result)
        p_score = prof_eval.get("score", 0)
        r_score = rand_eval.get("score", 0)
        print(f"    Q: {question[:60]}... | Prof={p_score} Rand={r_score}")

    return results


# ---------------------------------------------------------------------------
# Experiment 3b: Profile-Evidence Chain
# ---------------------------------------------------------------------------

CHAIN_SYSTEM_PROMPT = """You are analyzing how a user profile connects disparate pieces of evidence.

Given:
1. A question about a user
2. Multiple evidence messages from different parts of the conversation
3. The user's profile

Your task: Explain how the user profile TAGS connect these seemingly unrelated evidence messages into a coherent answer.

Output ONLY valid JSON:
{
  "connecting_tags": ["profile tags that connect the evidence"],
  "chain_explanation": "how these tags link the evidence together",
  "evidence_relevance": {"evidence_id": "why this evidence is relevant to the tags"}
}"""


def run_evidence_chain(
    llm: LLMClient,
    chat: Dict[str, Any],
    profile: Dict[str, Any],
    dia_map: Dict[str, Dict[str, Any]],
    max_qa: int = 5,
) -> List[Dict[str, Any]]:
    """Run profile-evidence chain analysis on multi-evidence QA pairs."""
    qa_pairs = chat.get("qa", [])
    # Focus on multi-evidence questions
    multi_ev_qa = [q for q in qa_pairs if len(q.get("evidence", [])) >= 3]
    random.seed(42)
    random.shuffle(multi_ev_qa)
    selected = multi_ev_qa[:max_qa]

    results = []
    profile_text = condense_profile_for_retrieval(profile)

    for qa in selected:
        question = qa["question"]
        evidence_ids = qa.get("evidence", [])

        # Gather evidence messages
        evidence_msgs = []
        for eid in evidence_ids:
            if eid in dia_map:
                evidence_msgs.append({
                    "id": eid,
                    "speaker": dia_map[eid]["speaker"],
                    "text": dia_map[eid]["text"],
                })

        if len(evidence_msgs) < 2:
            continue

        evidence_text = "\n".join(
            f"[{m['id']}] {m['speaker']}: {m['text']}" for m in evidence_msgs
        )

        prompt = (
            f"USER PROFILE:\n{profile_text}\n\n"
            f"QUESTION: {question}\n\n"
            f"EVIDENCE MESSAGES:\n{evidence_text}\n\n"
            f"Explain how the profile tags connect these evidence messages."
        )

        chain = _parse_json(llm.chat(
            CHAIN_SYSTEM_PROMPT, prompt,
            temperature=0.3, max_tokens=500,
        ))

        result = {
            "question": question,
            "ground_truth": qa["answer"],
            "num_evidence": len(evidence_msgs),
            "evidence_ids": evidence_ids,
            "chain_analysis": chain,
        }
        results.append(result)
        tags = chain.get("connecting_tags", [])
        print(f"    Q: {question[:50]}... | {len(tags)} connecting tags found")

    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_memory_experiments(
    dataset_dir: str = "dataset",
    output_dir: str = "data/memory_eval",
    max_qa_per_chat: int = 8,
    chat_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_files = sorted(dataset_dir.glob("Chat_*.json"))
    if chat_filter:
        chat_files = [f for f in chat_files if f.stem in chat_filter]

    print(f"[Memory] Processing {len(chat_files)} chat files")
    llm = LLMClient()

    all_probing: List[Dict[str, Any]] = []
    all_chains: List[Dict[str, Any]] = []

    for chat_file in chat_files:
        print(f"\n[Memory] Processing {chat_file.name}")
        chat = load_json(str(chat_file))
        user_speaker, _ = detect_speakers(chat)

        # Load profile
        profile_name = user_speaker.lower().replace(" ", "_")
        profile_path = dataset_dir / "output" / "user" / f"{profile_name}_profile.json"
        if not profile_path.exists():
            print(f"  [Skip] No profile for {user_speaker}")
            continue
        profile = load_json(str(profile_path))

        dia_map = build_evidence_map(chat)
        all_ids = get_all_message_ids(chat)

        print(f"  Running memory probing (3a)...")
        probing = run_memory_probing(llm, chat, profile, dia_map, all_ids, max_qa_per_chat)

        print(f"  Running evidence chain (3b)...")
        chains = run_evidence_chain(llm, chat, profile, dia_map, max_qa_per_chat // 2)

        all_probing.extend(probing)
        all_chains.extend(chains)

    # Save results
    results = {
        "probing_3a": all_probing,
        "evidence_chain_3b": all_chains,
        "config": {
            "max_qa_per_chat": max_qa_per_chat,
            "num_chats": len(chat_files),
        },
        "timestamp": datetime.now().isoformat(),
    }
    results_path = output_dir / "memory_results.json"
    save_json(str(results_path), results)
    print(f"\n[Memory] Results saved to {results_path}")

    # Aggregate
    summary = aggregate_memory_results(all_probing, all_chains)
    summary_path = output_dir / "memory_summary.json"
    save_json(str(summary_path), summary)
    print(f"[Memory] Summary saved to {summary_path}")
    return summary


def aggregate_memory_results(
    probing: List[Dict[str, Any]],
    chains: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Probing scores
    prof_scores = [p["profile_score"] for p in probing if isinstance(p.get("profile_score"), (int, float))]
    rand_scores = [p["random_score"] for p in probing if isinstance(p.get("random_score"), (int, float))]

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0

    # Win/tie/loss
    wins = ties = losses = 0
    for p in probing:
        ps = p.get("profile_score", 0)
        rs = p.get("random_score", 0)
        if isinstance(ps, (int, float)) and isinstance(rs, (int, float)):
            if ps > rs: wins += 1
            elif ps == rs: ties += 1
            else: losses += 1

    # Chain stats
    chain_tag_counts = [len(c.get("chain_analysis", {}).get("connecting_tags", [])) for c in chains]

    return {
        "probing_3a": {
            "num_questions": len(probing),
            "profile_avg_score": avg(prof_scores),
            "random_avg_score": avg(rand_scores),
            "improvement_delta": round(avg(prof_scores) - avg(rand_scores), 3),
            "improvement_pct": round((avg(prof_scores) - avg(rand_scores)) / max(avg(rand_scores), 0.001) * 100, 1),
            "win_tie_loss": f"{wins}/{ties}/{losses}",
        },
        "evidence_chain_3b": {
            "num_chains": len(chains),
            "avg_connecting_tags": avg(chain_tag_counts),
            "avg_evidence_per_chain": avg([c.get("num_evidence", 0) for c in chains]),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiments 3a/3b: Memory Association Validation"
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="data/memory_eval")
    parser.add_argument("--max-qa", type=int, default=8)
    parser.add_argument("--chats", nargs="*", default=None)
    args = parser.parse_args()

    run_memory_experiments(args.dataset_dir, args.output_dir, args.max_qa, args.chats)


if __name__ == "__main__":
    main()
