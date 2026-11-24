"""Discrete-event controller experiment, with an independent SQLite tool service.

The controller consumes observations, never scenario goals. All variants share
the same proposal corruption, tool faults, schema and task oracle.
"""
from __future__ import annotations

import heapq
import json
import sqlite3
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Policy:
    name: str
    memory: bool = False
    validated: bool = False
    playback: bool = False


POLICIES = {
    p.name: p for p in (
        Policy("baseline"), Policy("task_memory", memory=True),
        Policy("validated_execution", validated=True),
        Policy("playback_recovery", playback=True),
        Policy("combined", memory=True, validated=True, playback=True),
    )
}
FIELDS = {"customer_id", "appointment_id", "slot", "alternatives", "cancel"}


@dataclass
class Scenario:
    id: str
    family: str
    events: list[dict]
    expected_slot: str
    expected_status: str = "active"
    expect_communication: bool = True
    tool_fault: str = "none"
    blocked_slots: list[str] = field(default_factory=list)
    duration_ms: int = 0


class ToolService:
    """An isolated synthetic service. A receipt is the only execution authority."""

    def __init__(self, fault="none", blocked=()):
        self.db = sqlite3.connect(":memory:")
        self.db.executescript("""
            CREATE TABLE appointment(id TEXT PRIMARY KEY, customer TEXT, slot TEXT, status TEXT);
            INSERT INTO appointment VALUES ('a1','c1','original','active');
            CREATE TABLE receipt(key TEXT PRIMARY KEY, payload TEXT, response TEXT);
            CREATE TABLE audit(id INTEGER PRIMARY KEY, slot TEXT, status TEXT);
        """)
        self.fault, self.blocked = fault, set(blocked)
        self.calls = 0

    def execute(self, args: dict, key: str) -> dict:
        self.calls += 1
        payload = json.dumps(args, sort_keys=True)
        old = self.db.execute("SELECT payload,response FROM receipt WHERE key=?", (key,)).fetchone()
        if old:
            if old[0] != payload:
                return {"ok": False, "error": "idempotency_conflict"}
            return json.loads(old[1]) | {"replayed": True}
        if set(args) != {"customer_id", "appointment_id", "slot", "cancel"} or any(
            not isinstance(args.get(k), str) or not args[k] or len(args[k]) > 128
            for k in ("customer_id", "appointment_id", "slot")
        ) or type(args.get("cancel")) is not bool:
            return {"ok": False, "error": "schema"}
        record = self.db.execute("SELECT customer FROM appointment WHERE id=?", (args["appointment_id"],)).fetchone()
        if not record or record[0] != args["customer_id"]:
            return {"ok": False, "error": "identity"}
        if self.fault == "transient" and self.calls == 1:
            return {"ok": False, "error": "unavailable", "retryable": True}
        if args["slot"] in self.blocked:
            return {"ok": False, "error": "slot_unavailable"}
        status = "cancelled" if args["cancel"] else "active"
        response = {"ok": True, "slot": args["slot"], "status": status, "key": key}
        with self.db:
            self.db.execute("UPDATE appointment SET slot=?,status=? WHERE id=?", (args["slot"], status, args["appointment_id"]))
            self.db.execute("INSERT INTO audit(slot,status) VALUES (?,?)", (args["slot"], status))
            self.db.execute("INSERT INTO receipt VALUES (?,?,?)", (key, payload, json.dumps(response)))
        return response

    def state(self):
        slot, status = self.db.execute("SELECT slot,status FROM appointment WHERE id='a1'").fetchone()
        return {"slot": slot, "status": status}

    @property
    def commits(self):
        return self.db.execute("SELECT count(*) FROM audit").fetchone()[0]

    def close(self):
        self.db.close()


class Controller:
    def __init__(self, policy: Policy, service: ToolService):
        self.policy, self.service = policy, service
        self.context: deque[dict] = deque(maxlen=8)
        self.ledger: dict = {}
        # Revision register is shared transport bookkeeping, not planner memory.
        # It prevents repeated delivery of identical observations cancelling work.
        self.observed: dict = {}
        self.revision, self.sequence, self.now = 0, 0, 0
        self.queue: list[tuple] = []
        self.trace: list[dict] = []
        self.pending: dict[str, dict] = {}
        self.utterances: dict[str, dict] = {}
        self.last_receipt: dict | None = None
        self.heard: dict | None = None
        self.arguments_correct = self.argument_attempts = self.prevented = 0
        self.unsupported = self.recoveries = self.errors = self.stale_results = 0
        self.max_context = self.max_queue = 0

    def schedule(self, at: int, kind: str, **data):
        self.sequence += 1
        heapq.heappush(self.queue, (at, self.sequence, kind, data))
        self.max_queue = max(self.max_queue, len(self.queue))

    def log(self, kind, **data):
        self.trace.append({"at_ms": self.now, "kind": kind, **data})

    def facts(self):
        if self.policy.memory:
            return dict(self.ledger)
        facts = {}
        for item in self.context:
            facts.update({k: v for k, v in item.items() if k in FIELDS})
        return facts

    def canonical(self, facts):
        return {"customer_id": facts.get("customer_id"), "appointment_id": facts.get("appointment_id"),
                "slot": facts.get("slot"), "cancel": facts.get("cancel", False)}

    def dispatch(self, call):
        self.pending[call["key"]] = call
        self.schedule(self.now + call["execute_delay"], "execute", call=call)
        if call["receipt_delay"] > 500:
            self.schedule(self.now + call["execute_delay"] + 500, "timeout", call=call)

    def emit(self, receipt):
        uid = f"speech-{len(self.utterances)}"
        self.utterances[uid] = {"receipt": dict(receipt), "played": 0.0, "interrupted": False}
        self.log("generated", utterance=uid, receipt=receipt,
                 text=f"Appointment {receipt['status']} at {receipt['slot']}.")

    def handle(self, kind, data):
        self.log(kind, **data)
        if kind == "user":
            patch = data.get("facts", {})
            if not isinstance(patch, dict):
                raise ValueError("user facts must be an object")
            for key in ("customer_id", "appointment_id", "slot"):
                if key in patch and (not isinstance(patch[key], str) or not patch[key] or len(patch[key]) > 128):
                    raise ValueError(f"invalid user field: {key}")
            if "cancel" in patch and type(patch["cancel"]) is not bool:
                raise ValueError("cancel must be boolean")
            if "alternatives" in patch and (
                not isinstance(patch["alternatives"], list) or len(patch["alternatives"]) > 8 or
                any(not isinstance(v, str) or not v or len(v) > 128 for v in patch["alternatives"])
            ):
                raise ValueError("alternatives must be a bounded list of slot strings")
            self.context.append(dict(patch))
            confirmed = {k: v for k, v in patch.items() if k in FIELDS}
            if any(k not in self.observed or self.observed[k] != v for k, v in confirmed.items()):
                self.revision += 1
                if self.policy.playback:
                    # Invalidate queued confirmations for the superseded intent.
                    # Already rendered audio cannot be undone.
                    for utterance in self.utterances.values():
                        if utterance["played"] < 1:
                            utterance["interrupted"] = True
            self.observed.update(confirmed)
            if self.policy.memory:
                self.ledger.update(confirmed)
            self.max_context = max(self.max_context, len(self.context))
        elif kind == "propose":
            facts = self.facts()
            args = self.canonical(facts)
            # Fault injection is common to every policy, independent of the oracle.
            args.update(data.get("draft_patch", {}))
            if self.policy.validated:
                grounded = self.canonical(facts)
                if not all(isinstance(grounded[k], str) and grounded[k] for k in ("customer_id", "appointment_id", "slot")):
                    self.prevented += 1
                    self.log("needs_clarification")
                    return
                if args != grounded:
                    self.prevented += 1
                    self.log("repaired_draft", before=args, after=grounded)
                    args = grounded
            key = f"intent-{self.revision}" if self.policy.validated else f"attempt-{self.sequence}"
            self.dispatch({"args": args, "revision": self.revision, "key": key,
                           "execute_delay": data.get("execute_delay", 0),
                           "receipt_delay": data.get("receipt_delay", 0), "retry": 0})
        elif kind == "execute":
            call = data["call"]
            if self.policy.validated and call["revision"] != self.revision:
                self.prevented += 1
                self.pending.pop(call["key"], None)
                self.log("stale_execution_blocked")
                return
            current = self.canonical(self.facts())
            self.argument_attempts += 1
            self.arguments_correct += int(call["args"] == current and current["slot"] is not None)
            result = self.service.execute(call["args"], call["key"])
            self.schedule(self.now + call["receipt_delay"], "receipt", call=call, result=result)
        elif kind == "timeout":
            call = data["call"]
            if call["key"] not in self.pending or call["retry"] >= 1:
                return
            retry = dict(call, retry=call["retry"] + 1, execute_delay=0, receipt_delay=0)
            if not self.policy.validated:
                retry["key"] = f"retry-{self.sequence}"
            self.dispatch(retry)
        elif kind == "receipt":
            call, result = data["call"], data["result"]
            self.pending.pop(call["key"], None)
            if self.policy.validated and call["revision"] != self.revision:
                self.stale_results += 1
                return
            if result.get("ok"):
                if self.last_receipt and self.last_receipt.get("key") == result.get("key"):
                    return
                self.last_receipt = result
                self.emit(result)
            else:
                self.errors += 1
                if self.policy.validated and call["retry"] < 2:
                    if result.get("retryable"):
                        self.recoveries += 1
                        self.dispatch(dict(call, retry=call["retry"] + 1, execute_delay=50, receipt_delay=0))
                    elif result.get("error") == "slot_unavailable":
                        alternatives = self.facts().get("alternatives", [])
                        if alternatives:
                            # Alternatives were explicitly authorized in a user event.
                            slot = alternatives[0]
                            self.revision += 1
                            self.context.append({"slot": slot})
                            if self.policy.memory:
                                self.ledger["slot"] = slot
                            self.recoveries += 1
                            self.dispatch(dict(call, args=call["args"] | {"slot": slot},
                                               revision=self.revision, key=f"intent-{self.revision}",
                                               retry=call["retry"] + 1, execute_delay=0, receipt_delay=0))
        elif kind == "interrupt":
            for utterance in self.utterances.values():
                if utterance["played"] < 1:
                    utterance["interrupted"] = True
        elif kind == "repeat":
            if self.last_receipt:
                if self.policy.playback:
                    self.emit(self.last_receipt)
                else:
                    self.log("assumed_already_communicated")
        elif kind == "play":
            eligible = [(k, u) for k, u in self.utterances.items() if not u["interrupted"] and u["played"] < 1]
            if eligible:
                uid, utterance = eligible[0]
                fraction = data.get("fraction", 1.0)
                if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 <= fraction <= 1:
                    raise ValueError("playback fraction must be between zero and one")
                utterance["played"] = max(utterance["played"], fraction)
                if utterance["played"] == 1:
                    receipt = utterance["receipt"]
                    self.heard = {k: receipt[k] for k in ("slot", "status")}
                    if self.heard != self.service.state():
                        self.unsupported += 1
                    self.log("played", utterance=uid, receipt=receipt)
        elif kind != "tick":
            raise ValueError(f"unknown event: {kind}")

    def run(self, events):
        for event in events:
            event = dict(event)
            at, kind = event.pop("at"), event.pop("kind")
            if type(at) is not int or at < 0:
                raise ValueError("event time must be a nonnegative integer")
            self.schedule(at, kind, **event)
        steps = 0
        while self.queue:
            self.now, _, kind, data = heapq.heappop(self.queue)
            self.handle(kind, data)
            steps += 1
            if steps > 10000:
                raise RuntimeError("event budget exceeded")


def evaluate(scenario: Scenario, policy: Policy):
    service = ToolService(scenario.tool_fault, scenario.blocked_slots)
    controller = Controller(policy, service)
    try:
        controller.run(scenario.events)
        expected = {"slot": scenario.expected_slot, "status": scenario.expected_status}
        correct = service.state() == expected
        communicated = controller.heard == expected or not scenario.expect_communication
        # These scenarios require at most one committed mutation. Replays do not count.
        duplicates = service.db.execute(
            "SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM audit GROUP BY slot,status HAVING n>1)"
        ).fetchone()[0]
        return {
            "scenario": scenario.id, "family": scenario.family, "variant": policy.name,
            "passed": correct and communicated and not duplicates and not controller.unsupported,
            "state_correct": correct, "communication_correct": communicated,
            "arguments_correct": controller.arguments_correct, "argument_attempts": controller.argument_attempts,
            "duplicate_actions": duplicates, "unsupported_claims": controller.unsupported,
            "recoveries": controller.recoveries, "errors": controller.errors,
            "prevented": controller.prevented, "stale_results": controller.stale_results,
            "duration_ms": max(controller.now, scenario.duration_ms),
            "max_context": controller.max_context, "max_queue": controller.max_queue,
            "final_state": service.state(), "heard": controller.heard, "trace": controller.trace,
        }
    finally:
        service.close()
