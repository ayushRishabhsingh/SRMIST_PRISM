"""Vocabulary (from the official supported_classes.json) and natural-language prompts."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from .config import DATA_DIR, WEIGHTS_DIR


@lru_cache(maxsize=1)
def load_classes(path: str = "") -> dict:
    p = Path(path) if path else DATA_DIR / "supported_classes.json"
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d


# ---------------------------------------------------------------------------------------------
# Video-level labels (16 official labels). "other" exists in the vocabulary but is only a
# fallback: every evaluation video is either a kitchen (cooking) or one of the 14 FPVSum sports.
LABEL_DESCRIPTIONS: Dict[str, List[str]] = {
    "bike": ["mountain biking on a dirt trail", "riding a bicycle downhill", "cycling on a forest path"],
    "bikepolo": ["playing bike polo with a mallet", "a bicycle polo game on a hard court", "riding a bicycle and hitting a ball with a mallet"],
    "boxing": ["boxing sparring in a ring", "a boxer punching an opponent", "boxing gloves and a punching opponent"],
    "cooking": ["cooking in a kitchen", "preparing food on a kitchen counter", "washing dishes at a kitchen sink", "hands chopping vegetables in a kitchen"],
    "horse": ["riding a horse", "horseback riding on a trail", "a horse's head and mane seen by the rider"],
    "jumping": ["skydiving in free fall", "jumping out of an airplane with a parachute", "bungee jumping or base jumping from a height"],
    "longboarding": ["longboarding downhill on a road", "riding a longboard skateboard on asphalt", "fast skateboard ride down a mountain road"],
    "motor": ["riding a motorcycle", "motorbike riding on a road", "motocross riding on a dirt track"],
    "parkour": ["doing parkour", "running and jumping over walls and rooftops", "free running across urban obstacles"],
    "plane": ["flying a small airplane", "the view from an airplane cockpit", "piloting an aircraft above the clouds"],
    "rockclimbing": ["rock climbing on a cliff", "climbing a steep rock wall", "a climber's hands gripping rock holds"],
    "scuba": ["scuba diving underwater", "swimming underwater near a coral reef", "diving with fish in the ocean"],
    "skate": ["skateboarding at a skatepark", "doing skateboard tricks on a ramp", "skating on a half pipe"],
    "ski": ["skiing down a snowy slope", "skiing in fresh powder snow", "snowboarding down a mountain"],
    "surfing": ["surfing a wave", "riding a surfboard in the ocean", "paddling a surfboard toward breaking waves"],
    "other": ["an ordinary everyday first-person video", "a person walking around"],
}
LABEL_TEMPLATES = [
    "a first-person video of {}.",
    "a GoPro point-of-view shot of {}.",
    "{}, filmed with a head-mounted camera.",
]

# Highlight-worthy moments per category (positive) and generic dull moments (negative).
HIGHLIGHT_POSITIVE: Dict[str, List[str]] = {
    "bike": ["a mountain biker jumping over a ramp", "a fast steep downhill section on a bike", "riding over rocks and roots at high speed"],
    "bikepolo": ["hitting the ball with the mallet", "scoring a goal in bike polo", "players fighting for the ball"],
    "boxing": ["throwing a punch at the opponent", "an intense exchange of punches", "the opponent getting hit"],
    "horse": ["a horse galloping fast", "a horse jumping over an obstacle", "a beautiful landscape seen from horseback"],
    "jumping": ["free falling through the sky", "the parachute opening", "jumping out of the plane door", "a breathtaking aerial view of the ground"],
    "longboarding": ["a fast high-speed downhill turn", "sliding sideways on a longboard", "carving through a sharp corner"],
    "motor": ["a motorcycle leaning into a sharp turn", "overtaking at high speed", "a motocross jump"],
    "parkour": ["jumping between rooftops", "vaulting over a wall", "a big leap in the air"],
    "plane": ["the airplane taking off from the runway", "landing the airplane", "a spectacular aerial view of the landscape"],
    "rockclimbing": ["reaching for a difficult hold", "a dramatic view down the cliff", "reaching the top of the climb"],
    "scuba": ["a colorful coral reef with fish", "a big fish or turtle swimming close", "a shark or ray underwater"],
    "skate": ["landing a skateboard trick", "a skateboarder in the air above a ramp", "grinding a rail"],
    "ski": ["a skier jumping in the air", "fast skiing through deep powder", "a spectacular mountain panorama"],
    "surfing": ["riding inside a breaking wave", "standing up on the surfboard on a big wave", "a wave crashing over the surfer"],
    "cooking": ["hands cutting food on a chopping board", "pouring ingredients into a pan", "hands picking up and using a kitchen utensil",
                "opening the fridge or a cupboard", "washing a dish under the tap"],
    "other": ["an interesting moment", "an exciting action"],
}
HIGHLIGHT_GENERIC_POSITIVE = ["an exciting moment", "a spectacular action shot", "an impressive view"]
HIGHLIGHT_NEGATIVE = [
    "a blurry frame", "a dark underexposed frame", "a static view of the ground",
    "a person standing still and waiting", "an empty boring scene", "the camera pointing at the sky",
]
KITCHEN_INTERACTION_POS = [
    "close-up of hands manipulating an object", "hands holding food or a utensil",
    "hands working on a kitchen counter", "hands using a kitchen appliance",
]
KITCHEN_INTERACTION_NEG = [
    "a kitchen with no hands visible", "walking across the room", "looking at a wall", "a blurry frame",
]


def label_prompts() -> Tuple[List[str], List[str]]:
    """Returns (labels, flattened prompts) with prompts grouped per label in the same order."""
    labels = list(load_classes()["labels"])
    prompts = []
    for lab in labels:
        descs = LABEL_DESCRIPTIONS.get(lab, [lab])
        for d in descs:
            for t in LABEL_TEMPLATES:
                prompts.append(t.format(d))
    return labels, prompts


def label_prompt_groups() -> List[Tuple[str, List[str]]]:
    labels = list(load_classes()["labels"])
    out = []
    for lab in labels:
        descs = LABEL_DESCRIPTIONS.get(lab, [lab])
        out.append((lab, [t.format(d) for d in descs for t in LABEL_TEMPLATES]))
    return out


# ---------------------------------------------------------------------------------------------
# EPIC verb / noun phrasing
_GERUND_EXCEPTIONS = {
    "take": "taking", "put": "putting", "close": "closing", "cut": "cutting", "move": "moving", "remove": "removing",
    "shake": "shaking", "squeeze": "squeezing", "empty": "emptying", "flip": "flipping", "scrape": "scraping",
    "scrub": "scrubbing", "pat": "patting", "wrap": "wrapping", "look": "looking at", "rip": "ripping",
    "search": "searching for", "divide": "dividing", "set": "setting", "rub": "rubbing", "drop": "dropping",
    "slide": "sliding", "water": "watering", "coat": "coating", "transition": "moving between", "measure": "measuring",
    "increase": "increasing", "wait": "waiting for", "use": "using", "grate": "grating", "let-go": "letting go of",
    "stab": "stabbing", "serve": "serving", "choose": "choosing", "prepare": "preparing", "bake": "baking",
    "unfreeze": "defrosting", "turn-on": "turning on", "turn-off": "turning off", "turn-down": "turning down",
    "dry": "drying", "apply": "applying", "knead": "kneading", "smell": "smelling", "feel": "feeling",
    "form": "forming", "sprinkle": "sprinkling", "unscrew": "unscrewing", "screw": "screwing", "wear": "wearing",
    "unwrap": "unwrapping", "carry": "carrying",
    "other": "doing something with",
}


def verb_phrase(v: str) -> str:
    if v in _GERUND_EXCEPTIONS:
        return _GERUND_EXCEPTIONS[v]
    v2 = v.replace("-", " ")
    w = v2.split(" ")
    head = w[0]
    if head.endswith("e") and not head.endswith("ee"):
        head = head[:-1] + "ing"
    else:
        head = head + "ing"
    return " ".join([head] + w[1:])


def noun_phrase(n: str) -> str:
    if n == "other":
        return "an object"
    parts = n.split(":")
    if len(parts) == 1:
        return parts[0]
    return " ".join(parts[1:] + [parts[0]])


def verb_prompts(verbs: List[str]) -> List[str]:
    return [f"a first-person photo of hands {verb_phrase(v)} something in a kitchen" for v in verbs]


def noun_prompts(nouns: List[str]) -> List[str]:
    return [f"a first-person photo of a {noun_phrase(n)} in a kitchen" for n in nouns]


def pair_prompts(pairs: List[Tuple[str, str]]) -> List[str]:
    return [f"a first-person photo of hands {verb_phrase(v)} the {noun_phrase(n)}" for v, n in pairs]


# ---------------------------------------------------------------------------------------------
def load_pair_prior(min_count: int = 2) -> Tuple[List[Tuple[str, str]], List[float]]:
    """Verb-noun pair frequencies computed from the EPIC-KITCHENS-100 *train* split only
    (weights/epic_action_prior.json, built by code/build_prior.py). Falls back to a uniform
    prior over all verb x noun combinations restricted to frequent verbs/nouns if missing."""
    cls = load_classes()
    verbs = [v for v in cls["verbs"] if v != "other"]
    nouns = [n for n in cls["nouns"] if n != "other"]
    p = WEIGHTS_DIR / "epic_action_prior.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        vs, ns = set(verbs), set(nouns)
        pairs, counts = [], []
        for item in d["pairs"]:
            v, n, c = item["verb"], item["noun"], item["count"]
            if c >= min_count and v in vs and n in ns:
                pairs.append((v, n)); counts.append(float(c))
        if pairs:
            return pairs, counts
    # fallback: top-30 verbs x top-60 nouns (vocab is frequency-ordered in EPIC)
    pairs = [(v, n) for v in verbs[:30] for n in nouns[:60]]
    return pairs, [1.0] * len(pairs)
