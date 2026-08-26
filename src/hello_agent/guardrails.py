import re

_INJECTION_PATTERNS = [
    r"ignore (all )?(the )?previous instructions",
    r"disregard (the )?system prompt",
    r"forget (everything|all) (you (were|have been) told|above)",
    r"you are now (in )?(developer|dan|jailbreak) mode",
]

_COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in _INJECTION_PATTERNS
]


def detect_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _COMPILED_PATTERNS)
