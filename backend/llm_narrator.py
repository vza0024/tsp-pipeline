"""Generate student-friendly narration from Python-verified lesson facts.

The language model is responsible for wording only. Mathematical values and
simulation descriptions are created deterministically in Python before the model
is called.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

MODEL_ID = os.getenv("TSP_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("TSP_MAX_NEW_TOKENS", "350"))
TEMPERATURE = float(os.getenv("TSP_TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("TSP_TOP_P", "0.9"))

_pipeline: Any | None = None
_pipeline_lock = threading.Lock()


def _candidate_model_paths() -> list[Path]:
    """Return configured and conventional local model locations."""
    configured = os.getenv("TSP_MODEL_PATH")
    home = Path.home()
    candidates = [
        Path(configured).expanduser() if configured else None,
        home / "models" / "llama3.1-8b",
        home / "torch_env" / "llama3.1-8b",
        Path("/scratch/llama3.1-8b"),
        Path("/data/models/llama3.1-8b"),
    ]
    return [path for path in candidates if path is not None]


def _resolve_model_path() -> str:
    for path in _candidate_model_paths():
        if path.exists():
            LOGGER.info("Using local model at %s", path)
            return str(path)
    return MODEL_ID


def _get_pipeline() -> Any | None:
    """Load the text-generation pipeline once and reuse it across requests."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        try:
            import torch
            from transformers import pipeline
        except ImportError:
            LOGGER.exception(
                "Missing LLM dependencies. Install torch, transformers, and accelerate."
            )
            return None

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            LOGGER.info("Loading %s on %s", MODEL_ID, device)
            if torch.cuda.is_available():
                LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))
                LOGGER.info(
                    "VRAM: %s GB",
                    torch.cuda.get_device_properties(0).total_memory // 1024**3,
                )

            _pipeline = pipeline(
                "text-generation",
                model=_resolve_model_path(),
                dtype=dtype,
                device_map="auto",
            )
            LOGGER.info("Language model ready")
        except Exception:
            LOGGER.exception("Unable to load the language model")
            _pipeline = None

    return _pipeline


# ── CONSTANTS ─────────────────────────────────────────────────────────────────

ACTION_DESCRIPTIONS = {
    "bounce": "bounce upward one by one",
    "grouping": "slide into separate groups",
    "split": "split apart into halves",
    "push_apart": "get pushed away from the others",
    "fill": "fill in with color one by one",
    "stamp": "get stamped with an X mark",
    "rotate": "rotate and lift slightly",
    "grow": "grow larger in size",
    "lid_drop": "get sealed with a lid dropping from above",
    "spin": "spin around in place",
    "knock_over": "fall over sideways",
    "shrink": "shrink down to a smaller size",
    "open_book": "open up to reveal pages",
    "stack_books": "get extra copies stacked on top",
    "change_cover": "change their cover color",
    "stand_up": "rotate upright",
}

STEP_TITLES = {1: "💡 Think", 2: "🧱 See", 3: "⚡ Prove"}
SIM_STEPS = {1: "freeze", 2: "place", 3: "action"}


# ── PYTHON-DEFINED STEP FACTS ────────────────────────────────────────────────
# These functions produce EXACT descriptions of what Python computed
# and what CoppeliaSim will display. The LLM must only rephrase these.


def _build_step_facts(step_num, question, obj, action):
    """Return exact facts for each step. LLM must only rephrase these."""
    w = question["working"]
    q_text = question["question"]
    answer = question["answer"]
    qtype = question["qtype"]
    on, od = question["object_num"], question["object_den"]
    ga = question.get("group_a", 0)
    gb = question.get("group_b", 0)
    action_desc = ACTION_DESCRIPTIONS.get(action, action)

    if step_num == 1:
        return _step1_facts(qtype, q_text, answer, w, obj, od)
    elif step_num == 2:
        return _step2_facts(qtype, q_text, answer, w, obj, action, on, od, ga, gb)
    else:
        return _step3_facts(qtype, q_text, answer, w, obj, action, action_desc, on, od, ga, gb)


def _step1_facts(qtype, q_text, answer, w, obj, od):
    """Step 1: Screen is empty. Explain the concept and strategy."""

    if qtype in ("simplify", "equivalent"):
        ne, de = w.get("ne"), w.get("de")
        math_fact = f"To simplify {ne}/{de}, find the GCD of {ne} and {de}, then divide both by it."
    elif qtype == "compare":
        n1, n2, den = w.get("n1"), w.get("n2"), w.get("den")
        math_fact = f"To compare {n1}/{den} and {n2}/{den}, since they have the same denominator, just compare the numerators: {n1} vs {n2}."
    elif qtype == "add":
        n1, n2, den = w.get("n1"), w.get("n2"), w.get("den")
        math_fact = f"To add {n1}/{den} + {n2}/{den}, keep the denominator {den} and add the numerators: {n1} + {n2}."
    elif qtype in ("of", "dec_of"):
        whole = w.get("whole")
        num, den = w.get("num", w.get("sn")), w.get("den", w.get("sd"))
        math_fact = f"To find {num}/{den} of {whole}, divide {whole} into {den} equal groups, then take {num} of those groups."
    elif qtype in ("ratio_simplify", "ratio_equiv"):
        a, b = w.get("a"), w.get("b")
        math_fact = (
            f"To simplify the ratio {a}:{b}, find the GCD of {a} and {b}, then divide both by it."
        )
    elif qtype == "scale_up":
        sa, sb = w.get("sa"), w.get("sb")
        a = w.get("a")
        math_fact = f"The ratio is {sa}:{sb}. To scale up, find what {sa} was multiplied by to get {a}, then multiply {sb} by the same number."
    elif qtype == "part_whole":
        sa, sb = w.get("sa"), w.get("sb")
        total = w.get("total")
        math_fact = f"With ratio A:B = {sa}:{sb} and total {total}, divide {total} by ({sa}+{sb}) to find how many per part, then multiply."
    elif qtype == "unit_rate":
        qty, cost = w.get("qty"), w.get("cost")
        math_fact = (
            f"To find cost per item, divide the total cost ${cost} by the number of items {qty}."
        )
    elif qtype == "to_fraction":
        dec = w.get("dec")
        math_fact = f"To convert {dec} to a fraction, write it over the appropriate power of 10, then simplify."
    elif qtype == "to_percent":
        dec = w.get("dec")
        math_fact = f"To convert {dec} to a percentage, multiply by 100."
    elif qtype in ("dec_compare", "prob_compare"):
        d1 = w.get("d1", w.get("n1"))
        d2 = w.get("d2", w.get("n2"))
        math_fact = f"Compare {d1} and {d2} by looking at their values directly."
    elif qtype == "dec_add":
        d1, d2 = w.get("d1"), w.get("d2")
        math_fact = f"To add {d1} + {d2}, add the decimal values keeping the decimal point aligned."
    elif qtype == "of_number":
        pct, whole = w.get("pct"), w.get("whole")
        math_fact = f"To find {pct}% of {whole}, convert the percent to a fraction and multiply."
    elif qtype == "pct_fraction":
        pct = w.get("pct")
        math_fact = f"To convert {pct}% to a fraction, write {pct}/100 and simplify."
    elif qtype == "to_decimal":
        pct = w.get("pct")
        math_fact = f"To convert {pct}% to a decimal, divide by 100."
    elif qtype == "pct_change":
        orig, new_val = w.get("orig"), w.get("new_val")
        direction = w.get("direction")
        math_fact = f"To find the % {direction}, calculate the difference between {orig} and {new_val}, then divide by the original {orig}."
    elif qtype == "find_whole":
        part, pct = w.get("part"), w.get("pct")
        math_fact = f"{part} is {pct}% of the whole. To find the whole, divide {part} by {pct}/100."
    elif qtype == "simple":
        total, favorable = w.get("total"), w.get("favorable")
        color = w.get("color", "selected")
        math_fact = f"Probability = favorable outcomes / total outcomes = {favorable}/{total}."
    elif qtype == "complement":
        total, color_cnt = w.get("total"), w.get("color_cnt")
        color = w.get("color", "selected")
        not_cnt = w.get("not_cnt")
        math_fact = f"P(not {color}) = (total - {color} count) / total = ({total} - {color_cnt}) / {total} = {not_cnt}/{total}."
    elif qtype == "express":
        num, den = w.get("num"), w.get("den")
        fmt = w.get("fmt")
        math_fact = f"To express {num}/{den} as a {fmt}, divide {num} by {den}."
    elif qtype == "expected":
        num, den, trials = w.get("num"), w.get("den"), w.get("trials")
        math_fact = f"Expected = probability × trials = {num}/{den} × {trials}."
    else:
        math_fact = f"Solve: {q_text}"

    return {
        "visual": "The screen is empty — no objects yet.",
        "math": math_fact,
        "question": q_text,
        "answer": answer,
    }


def _step2_facts(qtype, q_text, answer, w, obj, action, on, od, ga, gb):
    """Step 2: Objects placed. Describe exactly what is visible."""

    # All objects placed neutral — no colors in Step 2
    visual = f"{od} {obj}s appear on the screen."

    # Describe the math calculation
    if qtype in ("simplify", "equivalent"):
        ne, de, gcd = w.get("ne"), w.get("de"), w.get("gcd")
        sn, sd = w.get("sn"), w.get("sd")
        calc = f"The {od} {obj}s represent the denominator {de}. The {on} colored ones represent the numerator {ne}. So we see {ne}/{de}. The GCD of {ne} and {de} is {gcd}. Dividing both: {ne}÷{gcd} = {sn}, {de}÷{gcd} = {sd}. So {ne}/{de} = {sn}/{sd}."
    elif qtype == "compare":
        n1, n2, den = w.get("n1"), w.get("n2"), w.get("den")
        bigger = w.get("bigger")
        calc = f"Both fractions have denominator {den}. Compare numerators: {n1} vs {n2}. Since {bigger} > {min(n1, n2)}, the answer is {bigger}/{den}."
    elif qtype == "add":
        n1, n2, den = w.get("n1"), w.get("n2"), w.get("den")
        total_n = w.get("total_n")
        calc = f"We add the numerators: {n1} + {n2} = {total_n}. The denominator stays {den}. So {n1}/{den} + {n2}/{den} = {total_n}/{den} = {answer}."
    elif qtype in ("of", "dec_of"):
        whole = w.get("whole")
        per_group = w.get("per_group")
        num = w.get("num", w.get("sn", on))
        den = w.get("den", w.get("sd", od))
        calc = f"The {od} {obj}s represent {od} equal groups. Each group has {per_group} items (since {whole} ÷ {od} = {per_group}). The {on} colored {obj}s show {num} groups. {num} × {per_group} = {answer}."
    elif qtype in ("ratio_simplify", "ratio_equiv"):
        a, b = w.get("a"), w.get("b")
        sa, sb = w.get("sa"), w.get("sb")
        gcd = w.get("gcd")
        calc = f"There are {ga} {obj}s in group A and {gb} in group B, showing ratio {a}:{b}. The GCD of {a} and {b} is {gcd}. Dividing both: {a}÷{gcd} = {sa}, {b}÷{gcd} = {sb}. So {a}:{b} = {sa}:{sb}."
    elif qtype == "scale_up":
        a, b = w.get("a"), w.get("b")
        sa, sb = w.get("sa"), w.get("sb")
        gcd = w.get("gcd")
        calc = f"The ratio {sa}:{sb} scaled by {gcd} gives {a}:{b}. {ga} {obj}s in group A, {gb} in group B. Answer is {answer}."
    elif qtype == "part_whole":
        sa, sb = w.get("sa"), w.get("sb")
        total = w.get("total")
        count_a = w.get("count_a")
        per_part = w.get("per_part")
        calc = f"Ratio A:B = {sa}:{sb}. Total parts = {sa}+{sb} = {sa + sb}. {total} ÷ {sa + sb} = {per_part} per part. A = {sa} × {per_part} = {count_a}. Answer is {answer}."
    elif qtype == "unit_rate":
        qty, cost, rate = w.get("qty"), w.get("cost"), w.get("rate")
        calc = f"The {od} {obj}s represent {qty} items. Total cost ${cost} ÷ {qty} items = ${rate} per item."
    elif qtype == "to_fraction":
        dec = w.get("dec")
        sn, sd = w.get("sn"), w.get("sd")
        calc = f"{dec} = {sn}/{sd} in simplest form."
    elif qtype == "to_percent":
        dec, pct = w.get("dec"), w.get("pct")
        calc = f"{dec} × 100 = {pct}%. The {on} colored {obj}s out of {od} show this proportion."
    elif qtype in ("dec_compare", "prob_compare"):
        d1 = w.get("d1", w.get("n1"))
        d2 = w.get("d2", w.get("n2"))
        bigger = w.get("bigger")
        den = w.get("den", od)
        calc = f"Comparing {d1} and {d2}: {bigger} is larger. The {on} colored {obj}s out of {od} show {answer}."
    elif qtype == "dec_add":
        d1, d2, total = w.get("d1"), w.get("d2"), w.get("total")
        calc = f"{d1} + {d2} = {total}. The {on} colored {obj}s out of {od} show this sum."
    elif qtype == "of_number":
        pct, whole = w.get("pct"), w.get("whole")
        per_group = w.get("per_group")
        result = w.get("result")
        on_f, od_f = w.get("on_f", on), w.get("od_f", od)
        calc = f"{pct}% = {on_f}/{od_f}. {whole} ÷ {od_f} = {per_group} per group. {on_f} × {per_group} = {result}."
    elif qtype == "pct_fraction":
        pct, sn, sd = w.get("pct"), w.get("sn"), w.get("sd")
        calc = f"{pct}% = {pct}/100. Simplify: {sn}/{sd}."
    elif qtype == "to_decimal":
        pct, val = w.get("pct"), w.get("val")
        calc = f"{pct}% ÷ 100 = {val}."
    elif qtype == "pct_change":
        orig, diff, pct = (
            w.get("orig"),
            w.get("diff"),
            w.get("pct"),
        )
        direction = w.get("direction")
        calc = f"Difference: {diff}. {diff}/{orig} × 100 = {pct}%. The {direction} is {pct}%."
    elif qtype == "find_whole":
        part, pct, whole = w.get("part"), w.get("pct"), w.get("whole")
        calc = f"{part} is {pct}% of the whole. {part} ÷ ({pct}/100) = {whole}."
    elif qtype == "simple":
        total, favorable = w.get("total"), w.get("favorable")
        calc = f"{favorable} favorable out of {total} total. P = {favorable}/{total}."
    elif qtype == "complement":
        total, not_cnt = w.get("total"), w.get("not_cnt")
        calc = f"{not_cnt} non-matching out of {total} total. P = {not_cnt}/{total}."
    elif qtype == "express":
        num, den, fmt, val = w.get("num"), w.get("den"), w.get("fmt"), w.get("val")
        calc = f"{num}/{den} = {val}{'%' if fmt == 'percent' else ''}."
    elif qtype == "expected":
        num, den, trials, exp = (
            w.get("num"),
            w.get("den"),
            w.get("trials"),
            w.get("exp"),
        )
        calc = f"P = {num}/{den}. Expected = {num}/{den} × {trials} = {exp}."
    else:
        calc = f"Answer is {answer}."

    return {
        "visual": visual,
        "calc": calc,
        "question": q_text,
        "answer": answer,
        "obj": obj,
        "count": od,
    }


def _step3_facts(qtype, q_text, answer, w, obj, action, action_desc, on, od, ga, gb):
    """Step 3: Action completed. Describe exactly what happened and why it proves the answer."""

    # Exact description of what CoppeliaSim just did
    if qtype in ("simplify", "equivalent"):
        gcd = w.get("gcd")
        ne, de = w.get("ne"), w.get("de")
        sn, sd = w.get("sn"), w.get("sd")
        if action == "grouping":
            num_groups = de // gcd if isinstance(gcd, int) and gcd > 0 else od
            active_groups = ne // gcd if isinstance(gcd, int) and gcd > 0 else on
            visual = f"The {od} {obj}s slid into {num_groups} equal groups of {gcd} each. {active_groups} groups are colored and {num_groups - active_groups} groups are gray."
            proof = f"{active_groups} colored groups out of {num_groups} total groups = {sn}/{sd}. This proves {ne}/{de} = {sn}/{sd}."
        else:
            visual = f"{on} of the {od} {obj}s {action_desc}."
            proof = f"The {on} active {obj}s out of {od} total = {ne}/{de}. Dividing both by GCD {gcd} gives {sn}/{sd} = {answer}."
    elif qtype == "compare":
        n1, n2, bigger = w.get("n1"), w.get("n2"), w.get("bigger")
        den = w.get("den")
        visual = f"The {on} colored {obj}s {action_desc}."
        proof = (
            f"{bigger}/{den} is bigger than {min(n1, n2)}/{den} because {bigger} > {min(n1, n2)}."
        )
    elif qtype == "add":
        n1, n2, total_n = w.get("n1"), w.get("n2"), w.get("total_n")
        den = w.get("den")
        visual = f"All {total_n} colored {obj}s (both groups) {action_desc}."
        proof = (
            f"{n1} + {n2} = {total_n} {obj}s out of {den}. So {n1}/{den} + {n2}/{den} = {answer}."
        )
    elif qtype in ("of", "dec_of"):
        per_group = w.get("per_group")
        whole = w.get("whole")
        visual = f"{on} of the {od} {obj}s {action_desc}."
        proof = f"Each {obj} represents a group of {per_group} (since {whole} ÷ {od} = {per_group}). {on} groups × {per_group} = {answer}."
    elif qtype in ("ratio_simplify", "ratio_equiv"):
        a, b = w.get("a"), w.get("b")
        sa, sb = w.get("sa"), w.get("sb")
        gcd = w.get("gcd")
        if action == "grouping":
            visual = f"The {obj}s slid into 2 groups: {ga} in group A and {gb} in group B."
            proof = f"Dividing both by {gcd}: {a}÷{gcd} = {sa}, {b}÷{gcd} = {sb}. So {a}:{b} = {sa}:{sb}."
        else:
            visual = f"{on} of the {od} {obj}s {action_desc}."
            proof = f"Group A ({ga}) and Group B ({gb}) show {a}:{b} = {sa}:{sb}."
    elif qtype == "scale_up":
        a, b = w.get("a"), w.get("b")
        sa, sb = w.get("sa"), w.get("sb")
        visual = f"{on} {obj}s in group A {action_desc}."
        proof = f"{sa}:{sb} scaled up gives {a}:{b}. Answer is {answer}."
    elif qtype == "part_whole":
        count_a = w.get("count_a", on)
        visual = f"{on} {obj}s in group A {action_desc}."
        proof = f"Group A has {count_a} {obj}s. Answer is {answer}."
    elif qtype == "unit_rate":
        qty, cost, rate = w.get("qty"), w.get("cost"), w.get("rate")
        visual = f"1 of the {od} {obj}s {action_desc}."
        proof = f"Each {obj} = 1 item. ${cost} ÷ {qty} = ${rate} per item."
    else:
        visual = f"{on} of the {od} {obj}s {action_desc}."
        proof = f"This shows the answer is {answer}."

    return {
        "visual": visual,
        "proof": proof,
        "question": q_text,
        "answer": answer,
        "obj": obj,
    }


# ── PROMPT ────────────────────────────────────────────────────────────────────


def _build_messages(step_num, question, obj, action, previous_steps):
    facts = _build_step_facts(step_num, question, obj, action)

    prev_text = ""
    if previous_steps:
        prev_text = "\nPREVIOUS STEPS:\n"
        for ps in previous_steps:
            prev_text += f"  Step {ps['step_index']}: {ps['instruction']}\n"

    if step_num == 1:
        user_msg = f"""Rephrase these facts as a friendly math explanation for a middle school student.

QUESTION: {facts["question"]}
ANSWER: {facts["answer"]}
WHAT IS ON SCREEN: {facts["visual"]}
MATH STRATEGY: {facts["math"]}
{prev_text}

RULES:
- "instruction": Restate the question in max 10 simple words.
- "explanation": 2 sentences. Sentence 1: what kind of problem is this. Sentence 2: the exact strategy from MATH STRATEGY above.
- Do NOT mention objects, colors, or the screen.
- Bold key numbers with <strong>.

JSON only:
{{"instruction": "...", "explanation": "..."}}"""

    elif step_num == 2:
        user_msg = f"""Rephrase these facts as a friendly math explanation for a middle school student.

QUESTION: {facts["question"]}
ANSWER: {facts["answer"]}
WHAT IS ON SCREEN: {facts["visual"]}
CALCULATION: {facts["calc"]}
{prev_text}

RULES:
- "instruction": Describe what the {facts.get("count", "")} {facts.get("obj", "")}s show in max 10 words.
- "explanation": 2-3 sentences. Rephrase the CALCULATION above in student-friendly language. Show the actual math steps. Must end with the answer: {facts["answer"]}.
- You MUST use ONLY the facts from WHAT IS ON SCREEN and CALCULATION. Do NOT invent any visual details.
- Bold key numbers with <strong>.

JSON only:
{{"instruction": "...", "explanation": "..."}}"""

    else:
        user_msg = f"""Rephrase these facts as a friendly math explanation for a middle school student.

QUESTION: {facts["question"]}
ANSWER: {facts["answer"]}
WHAT JUST HAPPENED: {facts["visual"]}
WHY THIS PROVES THE ANSWER: {facts["proof"]}
{prev_text}

RULES:
- "instruction": Describe what happened to the {facts.get("obj", "")}s in max 10 words. Use ONLY the description from WHAT JUST HAPPENED.
- "explanation": 2 sentences. Sentence 1: rephrase WHAT JUST HAPPENED. Sentence 2: rephrase WHY THIS PROVES THE ANSWER.
- You MUST use ONLY the facts provided. Do NOT invent details like "rows", "columns", "patterns" etc.
- Bold key numbers with <strong>.

JSON only:
{{"instruction": "...", "explanation": "..."}}"""

    return [
        {
            "role": "system",
            "content": "You are a friendly math tutor for middle schoolers. Respond with valid JSON only. Keep explanations SHORT (2-3 sentences max). Use simple words. You must ONLY rephrase the facts given — never invent visual descriptions. Never add text outside the JSON.",
        },
        {"role": "user", "content": user_msg},
    ]


# ── INFERENCE ─────────────────────────────────────────────────────────────────


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    text = raw_text.strip()

    if "```" in text:
        fenced_parts = [part.strip() for part in text.split("```")]
        json_parts = [
            part.removeprefix("json").strip()
            for part in fenced_parts
            if "{" in part and "}" in part
        ]
        if json_parts:
            text = json_parts[0]

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Expected a JSON object", text, start)
    return parsed


def _extract_quoted_field(text: str, field: str) -> str:
    """Recover a simple quoted JSON field from a malformed response."""
    pattern = rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    match = re.search(pattern, text)
    if not match:
        return ""
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return match.group(1)


def _sanitize_rich_text(value: Any) -> str:
    """Escape model output while preserving the supported ``strong`` tag."""
    escaped = html.escape(str(value or ""), quote=False)
    return escaped.replace("&lt;strong&gt;", "<strong>").replace("&lt;/strong&gt;", "</strong>")


def _normalize_result(result: Mapping[str, Any], fallback: Mapping[str, str]) -> dict[str, str]:
    instruction = _sanitize_rich_text(result.get("instruction"))
    explanation = _sanitize_rich_text(result.get("explanation"))
    return {
        "instruction": instruction or fallback["instruction"],
        "explanation": explanation or fallback["explanation"],
    }


def _call_llm(messages: Sequence[Mapping[str, str]], step_num: int) -> dict[str, str]:
    fallback = _fallback(messages, step_num)
    pipe = _get_pipeline()
    if pipe is None:
        return fallback

    raw = ""
    try:
        tokenizer = pipe.tokenizer
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        output = pipe(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=TEMPERATURE > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        raw = str(output[0]["generated_text"])
        if raw.startswith(prompt):
            raw = raw[len(prompt) :]

        parsed = _extract_json_object(raw)
        return _normalize_result(parsed, fallback)
    except json.JSONDecodeError:
        recovered = {
            "instruction": _extract_quoted_field(raw, "instruction"),
            "explanation": _extract_quoted_field(raw, "explanation"),
        }
        if recovered["instruction"] or recovered["explanation"]:
            LOGGER.warning("Recovered malformed model JSON for step %s", step_num)
            return _normalize_result(recovered, fallback)
        LOGGER.warning("Model returned unusable JSON for step %s", step_num)
    except Exception:
        LOGGER.exception("Narration inference failed for step %s", step_num)

    return fallback


def _prompt_value(content: str, label: str) -> str:
    prefix = f"{label}:"
    for line in content.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return ""


def _fallback(messages: Sequence[Mapping[str, str]], step_num: int) -> dict[str, str]:
    """Return deterministic narration when the model is unavailable."""
    content = messages[-1].get("content", "") if messages else ""
    question = _prompt_value(content, "QUESTION")
    answer = _prompt_value(content, "ANSWER")

    fallback_by_step = {
        1: {
            "instruction": "Let's understand this problem.",
            "explanation": (
                f"<strong>Question:</strong> {html.escape(question)} — identify the "
                "math concept and choose a strategy before calculating."
            ),
        },
        2: {
            "instruction": "Look at the objects and calculate.",
            "explanation": (
                "Use the displayed quantities to follow the calculation"
                + (f" toward <strong>{html.escape(answer)}</strong>." if answer else ".")
            ),
        },
        3: {
            "instruction": "Watch the result come to life.",
            "explanation": (
                f"The final animation supports the answer <strong>{html.escape(answer)}</strong>."
                if answer
                else "The final animation supports the computed result."
            ),
        },
    }
    return fallback_by_step.get(
        step_num,
        {"instruction": f"Step {step_num}.", "explanation": ""},
    )


# ── PUBLIC API ────────────────────────────────────────────────────────────────


def generate_step(
    step_num: int,
    question: Mapping[str, Any],
    obj: str,
    action: str,
    previous_steps: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate narration and metadata for one Think-See-Prove step."""
    if step_num not in STEP_TITLES:
        raise ValueError(f"step_num must be 1, 2, or 3; received {step_num}")

    messages = _build_messages(
        step_num,
        question,
        obj,
        action,
        list(previous_steps or []),
    )
    result = _call_llm(messages, step_num)

    object_num = int(question["object_num"])
    object_den = int(question["object_den"])
    return {
        "step_index": step_num,
        "instruction": result["instruction"],
        "explanation": result["explanation"],
        "sim_step": SIM_STEPS[step_num],
        "title": STEP_TITLES[step_num],
        "total": object_den,
        "highlighted": object_num if step_num == 3 else 0,
    }
