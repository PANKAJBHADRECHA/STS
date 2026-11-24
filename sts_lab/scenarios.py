"""Frozen scenario generator. Goal fields are consumed only by the evaluator.

Seeds vary identifiers/timing, not latent model behavior. This is an engineering
fault suite, not a representative sample of human conversations or tau-Voice.
"""
import random
from .core import Scenario

FAMILIES = ("argument_corrections", "tool_failures", "interruptions", "delayed_feedback", "long_sessions")
GENERATOR_VERSION = "1.1.0"


def event(at, kind, **data):
    return {"at": at, "kind": kind, **data}


def generate(split="held_out", seed=0):
    if split not in ("development", "held_out"):
        raise ValueError("split must be development or held_out")
    rng = random.Random((101 if split == "development" else 701) + seed)
    per_family = 6 if split == "development" else 20
    scenarios = []
    for family in FAMILIES:
        for i in range(per_family):
            subtype = i % 4
            day = rng.randrange(1, 29)
            slot = f"2026-12-{day:02d}T{rng.choice((9,10,11,14,15)):02d}:00"
            alternate = f"2026-12-{day:02d}T16:00"
            facts = {"customer_id": "c1", "appointment_id": "a1", "slot": slot}
            events = [event(0, "user", facts=facts)]
            expected, status, communicate, fault, blocked = slot, "active", True, "none", []
            jitter = rng.randrange(0, 10)
            if family == "argument_corrections":
                if subtype == 0:
                    events += [event(10, "user", facts={"slot": alternate}), event(20, "propose", draft_patch={"slot": slot})]
                    expected = alternate
                elif subtype == 1:
                    events += [event(20, "propose", draft_patch={"customer_id": "unconfirmed-customer"})]
                elif subtype == 2:
                    events[0] = event(0, "user", facts={"appointment_id": "a1", "slot": slot})
                    events += [event(10, "propose"), event(40, "user", facts={"customer_id": "c1"}), event(50, "propose")]
                else:
                    events += [event(10, "user", facts={"noise": "unrelated remark"}), event(20, "propose")]
                events += [event(300, "play")]
            elif family == "tool_failures":
                if subtype == 0:
                    fault = "transient"
                    events += [event(20, "propose")]
                elif subtype == 1:
                    events[0]["facts"]["alternatives"] = [alternate]
                    blocked, expected = [slot], alternate
                    events += [event(20, "propose")]
                elif subtype == 2:
                    events += [event(20, "propose", receipt_delay=1000+jitter)]
                else:
                    events += [event(20, "propose", draft_patch={"cancel": "false"})]
                events += [event(1500, "play")]
            elif family == "interruptions":
                if subtype == 0:
                    events += [event(10, "propose", execute_delay=100), event(20, "user", facts={"slot": alternate}), event(30, "interrupt"), event(150, "propose")]
                    expected = alternate
                elif subtype == 1:
                    events += [event(10, "propose"), event(20, "play", fraction=.3), event(30, "interrupt"), event(40, "repeat")]
                elif subtype == 2:
                    events += [event(10, "propose", execute_delay=100), event(20, "user", facts={"cancel": True}), event(30, "interrupt"), event(150, "propose")]
                    status = "cancelled"
                else:
                    events += [event(10, "propose", execute_delay=100), event(20, "user", facts={"backchannel": "mm-hmm"})]
                events += [event(400, "play")]
            elif family == "delayed_feedback":
                delay = (0, 300, 1000, 3000)[subtype]
                events += [event(10, "propose", receipt_delay=delay+jitter)]
                if subtype == 1:
                    events += [event(20, "user", facts={"slot": alternate}), event(30, "propose")]
                    expected = alternate
                elif subtype == 2:
                    events += [event(1600, "play", fraction=.5), event(1610, "interrupt"), event(1620, "repeat")]
                # Drain a second queued message too: do not hide stale output by
                # acknowledging only the newest/first correct confirmation.
                events += [event(4000, "play"), event(4010, "play")]
            else:
                # Thirty minutes of virtual time, not a wall-clock/model soak.
                for t in range(1000, 1_800_000, 4000):
                    events.append(event(t, "user", facts={"smalltalk": rng.choice(("weather", "travel", "music"))}))
                if subtype == 1:
                    events += [event(1_799_100, "user", facts={"slot": alternate})]
                    expected = alternate
                elif subtype == 2:
                    events += [event(1_799_100, "user", facts=dict(facts))]
                elif subtype == 3:
                    events += [event(1_799_100, "user", facts={"cancel": True})]
                    status = "cancelled"
                events += [event(1_799_200, "propose"), event(1_800_000, "play")]
            scenarios.append(Scenario(
                id=f"{split}-{family}-{i:02d}-seed{seed}", family=family, events=events,
                expected_slot=expected, expected_status=status, expect_communication=communicate,
                tool_fault=fault, blocked_slots=blocked,
                duration_ms=1_800_000 if family == "long_sessions" else 0,
            ))
    return scenarios
