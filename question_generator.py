"""
question_generator.py — Pure Python math only.

Imports: math, random — nothing else.
Every question includes a 'working' dict with ALL intermediate values
the LLM needs to write narration. No computation outside this file.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

from config import TOPICS, Question, validate_question

# ── HELPERS ───────────────────────────────────────────────────────────────────


def _q(
    topic: str,
    qtype: str,
    question: str,
    answer: object,
    object_num: int,
    object_den: int,
    group_a: int = 0,
    group_b: int = 0,
    **working: Any,
) -> Question:
    """Build and validate a question dictionary.

    ``None`` placeholders are removed from the working dictionary. All callers
    use ``dict.get`` for optional values, so omitting empty fields keeps prompts
    smaller without changing runtime behavior.
    """
    cleaned_working = {key: value for key, value in working.items() if value is not None}
    result: Question = {
        "topic": topic,
        "qtype": qtype,
        "question": question,
        "answer": str(answer),
        "object_num": object_num,
        "object_den": object_den,
        "group_a": group_a,
        "group_b": group_b,
        "working": cleaned_working,
    }
    validate_question(result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FRACTIONS
# ══════════════════════════════════════════════════════════════════════════════


def _fractions():
    qtype = random.choice(["simplify", "equivalent", "compare", "add", "of"])

    if qtype in ("simplify", "equivalent"):
        # Build from simplified form up, so GCD=k guaranteed
        sn = random.randint(1, 5)
        sd = random.randint(2, 7)
        while math.gcd(sn, sd) != 1 or sn >= sd:
            sd = random.randint(2, 7)
        max_k = min(5, 21 // sd)
        k = random.randint(2, max(2, max_k))
        ne = sn * k
        de = sd * k
        # Object display: use ne/de if de≤21 else sn/sd
        if de <= 21:
            on, od = ne, de
        else:
            on, od = sn, sd
        if qtype == "simplify":
            return _q(
                "fractions",
                "simplify",
                f"Simplify {ne}/{de} to its lowest terms.",
                f"{sn}/{sd}",
                on,
                od,
                ne=ne,
                de=de,
                sn=sn,
                sd=sd,
                gcd=k,
                per_group=None,
                whole=None,
            )
        else:
            return _q(
                "fractions",
                "equivalent",
                f"Show that {ne}/{de} = {sn}/{sd}.",
                f"{sn}/{sd}",
                on,
                od,
                ne=ne,
                de=de,
                sn=sn,
                sd=sd,
                gcd=k,
                per_group=None,
                whole=None,
            )

    elif qtype == "compare":
        den = random.randint(3, 12)
        n1 = random.randint(1, den - 1)
        n2 = random.randint(1, den - 1)
        while n1 == n2:
            n2 = random.randint(1, den - 1)
        bigger = max(n1, n2)
        od = min(den, 21)
        on = bigger
        # group_a = bigger fraction count (col_a), rest are col_b for smaller
        # This lets the sim show both fractions visually
        return _q(
            "fractions",
            "compare",
            f"Which is bigger: {n1}/{den} or {n2}/{den}?",
            f"{bigger}/{den}",
            on,
            od,
            group_a=bigger,
            group_b=od - bigger,
            n1=n1,
            n2=n2,
            den=den,
            bigger=bigger,
            ne=None,
            de=None,
            sn=None,
            sd=None,
            gcd=None,
            per_group=None,
            whole=None,
        )

    elif qtype == "add":
        den = random.randint(3, 10)
        n1 = random.randint(1, den - 1)
        n2 = random.randint(1, den - n1)
        total_n = n1 + n2
        g = math.gcd(total_n, den)
        sn = total_n // g
        sd = den // g
        od = min(den, 21)
        on = min(total_n, od)
        # group_a = n1 (first addend in col_a), next n2 in col_b, rest neutral
        # This lets the sim show n1 + n2 visually
        return _q(
            "fractions",
            "add",
            f"What is {n1}/{den} + {n2}/{den}?",
            f"{sn}/{sd}",
            on,
            od,
            group_a=n1,
            group_b=n2,
            n1=n1,
            n2=n2,
            den=den,
            total_n=total_n,
            sn=sn,
            sd=sd,
            gcd=g,
            ne=None,
            de=None,
            per_group=None,
            whole=None,
        )

    else:  # of
        num = random.randint(1, 4)
        den = random.randint(2, 6)
        while num >= den:
            num = random.randint(1, 4)
        whole = den * random.randint(2, 8)
        per_group = whole // den
        result = num * per_group
        return _q(
            "fractions",
            "of",
            f"What is {num}/{den} of {whole}?",
            str(result),
            num,
            den,
            num=num,
            den=den,
            whole=whole,
            per_group=per_group,
            result=result,
            ne=None,
            de=None,
            sn=None,
            sd=None,
            gcd=None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# RATIOS
# ══════════════════════════════════════════════════════════════════════════════


def _ratios():
    qtype = random.choice(["ratio_simplify", "ratio_equiv", "scale_up", "part_whole", "unit_rate"])

    if qtype in ("ratio_simplify", "ratio_equiv"):
        # sa:sb coprime, k≥2, total=(sa+sb)*k ≤ 21
        a = b = sa = sb = k = None
        for _ in range(50):
            sa = random.randint(1, 5)
            sb = random.randint(1, 5)
            if math.gcd(sa, sb) != 1 or sa == sb:
                continue
            max_k = 21 // (sa + sb)
            if max_k < 2:
                continue
            k = random.randint(2, max_k)
            a = sa * k
            b = sb * k
            if a + b <= 21:
                break
        if a is None:
            # Fallback: safe default
            sa, sb, k = 1, 2, 3
            a, b = 3, 6
        total = a + b
        if qtype == "ratio_simplify":
            return _q(
                "ratios",
                "ratio_simplify",
                f"Simplify the ratio {a}:{b}.",
                f"{sa}:{sb}",
                a,
                total,
                group_a=a,
                group_b=b,
                a=a,
                b=b,
                sa=sa,
                sb=sb,
                gcd=k,
                total=total,
                per_part=None,
                count_a=None,
                qty=None,
                rate=None,
                cost=None,
            )
        else:
            return _q(
                "ratios",
                "ratio_equiv",
                f"Show that {a}:{b} is the same ratio as {sa}:{sb}.",
                f"{sa}:{sb}",
                sa,
                sa + sb,
                group_a=sa,
                group_b=sb,
                a=a,
                b=b,
                sa=sa,
                sb=sb,
                gcd=k,
                total=sa + sb,
                per_part=None,
                count_a=None,
                qty=None,
                rate=None,
                cost=None,
            )

    elif qtype == "scale_up":
        sa = random.randint(1, 5)
        sb = random.randint(1, 5)
        while sa == sb:
            sb = random.randint(1, 5)
        max_k = 21 // (sa + sb)
        k = random.randint(2, max(2, max_k))
        a = sa * k
        b = sb * k
        if a + b > 21:
            k = max(1, 21 // (sa + sb))
            a = sa * k
            b = sb * k
        return _q(
            "ratios",
            "scale_up",
            f"If the ratio is {sa}:{sb}, what is {a}:?",
            str(b),
            a,
            a + b,
            group_a=a,
            group_b=b,
            a=a,
            b=b,
            sa=sa,
            sb=sb,
            gcd=k,
            total=a + b,
            per_part=None,
            count_a=None,
            qty=None,
            rate=None,
            cost=None,
        )

    elif qtype == "part_whole":
        sa = sb = mult = total = None
        for _ in range(50):
            sa = random.randint(1, 4)
            sb = random.randint(1, 4)
            if sa == sb:
                continue
            parts = sa + sb
            max_m = 21 // parts
            if max_m < 2:
                continue
            mult = random.randint(2, max_m)
            total = parts * mult
            if total <= 21:
                break
        if total is None:
            # Fallback: safe default
            sa, sb, mult = 1, 2, 3
            total = 9
        count_a = sa * mult
        per_part = mult
        return _q(
            "ratios",
            "part_whole",
            f"In a group of {total}, ratio A:B = {sa}:{sb}. How many are A?",
            str(count_a),
            count_a,
            total,
            group_a=count_a,
            group_b=total - count_a,
            sa=sa,
            sb=sb,
            total=total,
            count_a=count_a,
            per_part=per_part,
            gcd=mult,
            a=None,
            b=None,
            qty=None,
            rate=None,
            cost=None,
        )

    else:  # unit_rate
        rate = random.randint(2, 15)
        qty = random.randint(2, 8)
        cost = rate * qty
        return _q(
            "ratios",
            "unit_rate",
            f"If {qty} items cost ${cost}, what is the cost per item?",
            f"${rate}",
            1,
            qty,
            qty=qty,
            rate=rate,
            cost=cost,
            sa=None,
            sb=None,
            total=None,
            count_a=None,
            per_part=None,
            gcd=None,
            a=None,
            b=None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# DECIMALS
# ══════════════════════════════════════════════════════════════════════════════


def _decimals():
    qtype = random.choice(["to_fraction", "to_percent", "dec_compare", "dec_add", "dec_of"])

    if qtype == "to_fraction":
        # Clean decimals: tenths, quarters, fifths, eighths, twentieths
        options = [
            (1, 2),
            (1, 4),
            (3, 4),
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (1, 8),
            (3, 8),
            (5, 8),
            (7, 8),
            (1, 10),
            (3, 10),
            (7, 10),
            (9, 10),
            (1, 20),
            (3, 20),
            (7, 20),
            (9, 20),
            (11, 20),
            (13, 20),
        ]
        sn, sd = random.choice(options)
        g = math.gcd(sn, sd)
        sn2 = sn // g
        sd2 = sd // g
        dec = round(sn / sd, 4)
        return _q(
            "decimals",
            "to_fraction",
            f"What fraction is {dec} in simplest form?",
            f"{sn2}/{sd2}",
            sn2,
            sd2,
            dec=dec,
            num=sn,
            den=sd,
            sn=sn2,
            sd=sd2,
            gcd=g,
            d1=None,
            d2=None,
            whole=None,
            per_group=None,
            result=None,
        )

    elif qtype == "to_percent":
        options = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
        dec = random.choice(options)
        pct = int(dec * 100)
        # denominator: use 10 if dec is tenth, else 4 or 5
        if pct % 10 == 0:
            od = 10
            on = pct // 10
        elif pct % 25 == 0:
            od = 4
            on = pct // 25
        else:
            od = 5
            on = pct // 20
        on = min(on, od)
        return _q(
            "decimals",
            "to_percent",
            f"Convert {dec} to a percentage.",
            f"{pct}%",
            on,
            od,
            dec=dec,
            pct=pct,
            d1=None,
            d2=None,
            whole=None,
            per_group=None,
            result=None,
            num=None,
            den=None,
            sn=None,
            sd=None,
            gcd=None,
        )

    elif qtype == "dec_compare":
        d1 = round(random.randint(1, 9) * 0.1, 1)
        d2 = round(random.randint(1, 9) * 0.1, 1)
        while d1 == d2:
            d2 = round(random.randint(1, 9) * 0.1, 1)
        bigger = max(d1, d2)
        t1 = int(d1 * 10)
        t2 = int(d2 * 10)
        on = int(bigger * 10)
        return _q(
            "decimals",
            "dec_compare",
            f"Which is bigger: {d1} or {d2}?",
            str(bigger),
            on,
            10,
            d1=d1,
            d2=d2,
            t1=t1,
            t2=t2,
            bigger=bigger,
            dec=None,
            pct=None,
            whole=None,
            per_group=None,
            result=None,
            num=None,
            den=None,
            sn=None,
            sd=None,
            gcd=None,
        )

    elif qtype == "dec_add":
        d1 = round(random.randint(1, 8) * 0.1, 1)
        d2 = round(random.randint(1, 9 - int(d1 * 10)) * 0.1, 1)
        total = round(d1 + d2, 1)
        on = int(total * 10)
        return _q(
            "decimals",
            "dec_add",
            f"What is {d1} + {d2}?",
            str(total),
            on,
            10,
            d1=d1,
            d2=d2,
            total=total,
            dec=None,
            pct=None,
            whole=None,
            per_group=None,
            result=None,
            num=None,
            den=None,
            sn=None,
            sd=None,
            gcd=None,
        )

    else:  # dec_of
        options = [(1, 2), (1, 4), (3, 4), (1, 5), (2, 5), (1, 10), (3, 10)]
        sn, sd = random.choice(options)
        dec = round(sn / sd, 4)
        whole = sd * random.randint(2, 6)
        per_group = whole // sd
        result = sn * per_group
        return _q(
            "decimals",
            "dec_of",
            f"What is {dec} of {whole}?",
            str(result),
            sn,
            sd,
            dec=dec,
            num=sn,
            den=sd,
            whole=whole,
            per_group=per_group,
            result=result,
            d1=None,
            d2=None,
            sn=None,
            sd=None,
            gcd=None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PERCENTAGES
# ══════════════════════════════════════════════════════════════════════════════


def _percentages():
    qtype = random.choice(["of_number", "pct_fraction", "to_decimal", "pct_change", "find_whole"])

    if qtype == "of_number":
        # pct = on/od * 100, whole divisible by od
        fracs = [
            (1, 2, 50),
            (1, 4, 25),
            (3, 4, 75),
            (1, 5, 20),
            (2, 5, 40),
            (3, 5, 60),
            (4, 5, 80),
            (1, 10, 10),
            (3, 10, 30),
            (7, 10, 70),
            (9, 10, 90),
        ]
        on_f, od_f, pct = random.choice(fracs)
        whole = od_f * random.randint(2, 21 // od_f)
        per_group = whole // od_f
        result = on_f * per_group
        return _q(
            "percentages",
            "of_number",
            f"What is {pct}% of {whole}?",
            str(result),
            on_f,
            od_f,
            pct=pct,
            whole=whole,
            per_group=per_group,
            result=result,
            on_f=on_f,
            od_f=od_f,
            sn=None,
            sd=None,
            gcd=None,
            orig=None,
            new_val=None,
            diff=None,
            direction=None,
            unit=None,
            part=None,
        )

    elif qtype == "pct_fraction":
        pcts = [
            (50, 1, 2, 50),
            (25, 1, 4, 25),
            (75, 3, 4, 25),
            (20, 1, 5, 20),
            (40, 2, 5, 20),
            (10, 1, 10, 10),
            (30, 3, 10, 10),
            (60, 3, 5, 20),
            (80, 4, 5, 20),
            (90, 9, 10, 10),
        ]
        pct, sn, sd, gcd_v = random.choice(pcts)
        return _q(
            "percentages",
            "pct_fraction",
            f"Convert {pct}% to a fraction in simplest form.",
            f"{sn}/{sd}",
            sn,
            sd,
            pct=pct,
            sn=sn,
            sd=sd,
            gcd=gcd_v,
            whole=None,
            per_group=None,
            result=None,
            on_f=None,
            od_f=None,
            orig=None,
            new_val=None,
            diff=None,
            direction=None,
            unit=None,
            part=None,
        )

    elif qtype == "to_decimal":
        pcts = [
            (50, 0.5, 2, 1, 10),
            (25, 0.25, 4, 1, 4),
            (75, 0.75, 4, 3, 4),
            (20, 0.2, 10, 2, 10),
            (40, 0.4, 10, 4, 10),
            (10, 0.1, 10, 1, 10),
            (30, 0.3, 10, 3, 10),
            (60, 0.6, 10, 6, 10),
            (80, 0.8, 10, 8, 10),
        ]
        pct, val, od, on, _ = random.choice(pcts)
        return _q(
            "percentages",
            "to_decimal",
            f"Convert {pct}% to a decimal.",
            str(val),
            on,
            od,
            pct=pct,
            val=val,
            whole=None,
            per_group=None,
            result=None,
            on_f=None,
            od_f=None,
            sn=None,
            sd=None,
            gcd=None,
            orig=None,
            new_val=None,
            diff=None,
            direction=None,
            unit=None,
            part=None,
        )

    elif qtype == "pct_change":
        orig = random.choice([10, 20, 25, 40, 50, 60, 80, 100, 120, 200])
        # pick a clean % change
        pct_opts = [10, 20, 25, 50, 100]
        pct = random.choice(pct_opts)
        diff = orig * pct // 100
        direction = random.choice(["increase", "decrease"])
        new_val = orig + diff if direction == "increase" else orig - diff
        # objects: od = 100//pct groups, on = diff//unit
        od = 100 // pct
        on = 1
        unit = orig // od
        return _q(
            "percentages",
            "pct_change",
            f"A value goes from {orig} to {new_val}. What is the % {direction}?",
            f"{pct}%",
            on,
            od,
            orig=orig,
            new_val=new_val,
            diff=diff,
            direction=direction,
            pct=pct,
            unit=unit,
            whole=None,
            per_group=None,
            result=None,
            on_f=None,
            od_f=None,
            sn=None,
            sd=None,
            gcd=None,
            part=None,
        )

    else:  # find_whole
        fracs = [
            (1, 2, 50),
            (1, 4, 25),
            (1, 5, 20),
            (1, 10, 10),
            (3, 4, 75),
            (2, 5, 40),
        ]
        on_f, od_f, pct = random.choice(fracs)
        per_group = random.randint(2, 10)
        part = on_f * per_group
        whole = od_f * per_group
        return _q(
            "percentages",
            "find_whole",
            f"{part} is {pct}% of what number?",
            str(whole),
            on_f,
            od_f,
            pct=pct,
            part=part,
            whole=whole,
            per_group=per_group,
            on_f=on_f,
            od_f=od_f,
            result=None,
            sn=None,
            sd=None,
            gcd=None,
            orig=None,
            new_val=None,
            diff=None,
            direction=None,
            unit=None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PROBABILITY
# ══════════════════════════════════════════════════════════════════════════════


def _probability():
    qtype = random.choice(["simple", "complement", "express", "expected", "prob_compare"])

    if qtype == "simple":
        total = random.randint(4, 20)
        color = random.choice(["red", "blue", "green", "yellow", "purple"])
        favorable = random.randint(1, total - 1)
        g = math.gcd(favorable, total)
        sn = favorable // g
        sd = total // g
        return _q(
            "probability",
            "simple",
            f"A bag has {total} balls. {favorable} are {color}. What is P({color})?",
            f"{favorable}/{total}",
            favorable,
            total,
            total=total,
            color=color,
            favorable=favorable,
            sn=sn,
            sd=sd,
            gcd=g,
            color_cnt=None,
            not_cnt=None,
            num=None,
            den=None,
            fmt=None,
            trials=None,
            exp=None,
            n1=None,
            n2=None,
        )

    elif qtype == "complement":
        total = random.randint(5, 15)
        color = random.choice(["red", "blue", "green"])
        color_cnt = random.randint(1, total - 2)
        not_cnt = total - color_cnt
        g = math.gcd(not_cnt, total)
        sn = not_cnt // g
        sd = total // g
        return _q(
            "probability",
            "complement",
            f"A bag has {total} balls. {color_cnt} are {color}. What is P(not {color})?",
            f"{not_cnt}/{total}",
            not_cnt,
            total,
            total=total,
            color=color,
            color_cnt=color_cnt,
            not_cnt=not_cnt,
            sn=sn,
            sd=sd,
            gcd=g,
            favorable=None,
            num=None,
            den=None,
            fmt=None,
            trials=None,
            exp=None,
            n1=None,
            n2=None,
        )

    elif qtype == "express":
        options = [
            (1, 2, "decimal", 0.5),
            (1, 4, "decimal", 0.25),
            (3, 4, "decimal", 0.75),
            (1, 5, "decimal", 0.2),
            (1, 2, "percent", 50),
            (1, 4, "percent", 25),
            (3, 4, "percent", 75),
            (1, 5, "percent", 20),
            (2, 5, "percent", 40),
        ]
        num, den, fmt, val = random.choice(options)
        ans = str(val) + ("%" if fmt == "percent" else "")
        return _q(
            "probability",
            "express",
            f"Express probability {num}/{den} as a {fmt}.",
            ans,
            num,
            den,
            num=num,
            den=den,
            fmt=fmt,
            val=val,
            total=None,
            color=None,
            favorable=None,
            color_cnt=None,
            not_cnt=None,
            sn=None,
            sd=None,
            gcd=None,
            trials=None,
            exp=None,
            n1=None,
            n2=None,
        )

    elif qtype == "expected":
        den = random.choice([2, 4, 5])
        num = random.randint(1, den - 1)
        max_m = 21 // den
        mult = random.randint(2, max(2, max_m))
        trials = den * mult
        if trials > 21:
            trials = den * 2
        exp = (num * trials) // den
        return _q(
            "probability",
            "expected",
            f"If P = {num}/{den} and you repeat {trials} times, how many successes expected?",
            str(exp),
            exp,
            trials,
            num=num,
            den=den,
            trials=trials,
            exp=exp,
            total=None,
            color=None,
            favorable=None,
            color_cnt=None,
            not_cnt=None,
            sn=None,
            sd=None,
            gcd=None,
            fmt=None,
            n1=None,
            n2=None,
        )

    else:  # prob_compare
        den = random.randint(4, 10)
        n1 = random.randint(1, den - 1)
        n2 = random.randint(1, den - 1)
        while n1 == n2:
            n2 = random.randint(1, den - 1)
        bigger = max(n1, n2)
        return _q(
            "probability",
            "prob_compare",
            f"Which is more likely: {n1}/{den} or {n2}/{den}?",
            f"{bigger}/{den}",
            bigger,
            den,
            n1=n1,
            n2=n2,
            den=den,
            bigger=bigger,
            total=None,
            color=None,
            favorable=None,
            color_cnt=None,
            not_cnt=None,
            sn=None,
            sd=None,
            gcd=None,
            fmt=None,
            trials=None,
            exp=None,
        )


# ── PUBLIC API ─────────────────────────────────────────────────────────────────
_GENERATORS: dict[str, Callable[[], Question]] = {
    "fractions": _fractions,
    "ratios": _ratios,
    "decimals": _decimals,
    "percentages": _percentages,
    "probability": _probability,
}


def generate(topic: str) -> Question:
    """Generate and validate one question for a supported topic."""
    try:
        generator = _GENERATORS[topic]
    except KeyError as exc:
        supported = ", ".join(TOPICS)
        raise ValueError(f"Unsupported topic {topic!r}. Choose from: {supported}") from exc

    question = generator()
    validate_question(question)
    return question
