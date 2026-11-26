"""Strict, disclosed regression checks, separate from production inference."""

import re

WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]
MATH = {
    "normal": 4,
    "changed_question": 5,
    "background_noise": 8,
    "slow_speech": 7,
    "rapid_1": 5,
    "rapid_2": 7,
    "rapid_3": 9,
    "spoken_interruption": 6,
}


def answer_number(text):
    value = re.sub(r"[*_`]", "", text.lower())
    for n, word in enumerate(WORDS):
        value = re.sub(r"\b" + word + r"\b", str(n), value)
    results = re.findall(r"=\s*(-?\d+(?:\.\d+)?)", value)
    if not results:
        results = re.findall(
            r"\b(?:equals?|is|are|makes?|gives?|it's)\s+(?:exactly\s+)?(-?\d+(?:\.\d+)?)",
            value,
        )
    if results:
        return float(results[-1])
    first = re.match(r"^\s*(-?\d+(?:\.\d+)?)\b", value)
    if first:
        return float(first[1])
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", value)
    return float(nums[0]) if len(nums) == 1 else None


def check(name, text, pattern=""):
    if name in MATH:
        return answer_number(text) == MATH[name]
    if name == "memory" and re.search(
        r"\b(don't|cannot|can't|do not|not remember|not know|not have)\b", text.lower()
    ):
        return False
    return bool(re.search(pattern, text.lower())) and not bool(
        re.search(
            r"\bnot\b(?:\W+\w+){0,3}\W+(alex|london|friday|paris)\b", text.lower()
        )
    )
