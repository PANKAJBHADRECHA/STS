import random
import unittest

from sts_lab.core import Controller, POLICIES, Scenario, ToolService, evaluate
from sts_lab.scenarios import event, generate


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.service = ToolService()
        self.args = {"customer_id": "c1", "appointment_id": "a1", "slot": "noon", "cancel": False}

    def tearDown(self):
        self.service.close()

    def test_idempotent_retry_has_one_commit(self):
        self.assertTrue(self.service.execute(self.args, "key")["ok"])
        self.assertTrue(self.service.execute(self.args, "key")["replayed"])
        self.assertEqual(self.service.commits, 1)

    def test_key_cannot_be_reused_with_new_arguments(self):
        self.service.execute(self.args, "key")
        self.assertEqual(self.service.execute(self.args | {"slot": "night"}, "key")["error"], "idempotency_conflict")
        self.assertEqual(self.service.state()["slot"], "noon")

    def test_wrong_identity_is_rejected(self):
        self.assertFalse(self.service.execute(self.args | {"customer_id": "c2"}, "key")["ok"])
        self.assertEqual(self.service.commits, 0)

    def test_malformed_and_extra_fields_do_not_commit(self):
        for patch in ({"cancel": "false"}, {"slot": None}, {"slot": ""}, {"slot": 9}, {"extra": True}, {"slot": "x"*129}):
            with self.subTest(patch=patch):
                self.assertEqual(self.service.execute(self.args | patch, "key")["error"], "schema")
        self.assertEqual(self.service.commits, 0)

    def test_sql_text_is_data(self):
        self.assertTrue(self.service.execute(self.args | {"slot": "'; DROP TABLE appointment; --"}, "key")["ok"])
        self.assertEqual(self.service.commits, 1)


class ControllerTests(unittest.TestCase):
    def run_case(self, events, **kwargs):
        return evaluate(Scenario("test", "test", events, **kwargs), POLICIES["combined"])

    def initial(self):
        return event(0, "user", facts={"customer_id": "c1", "appointment_id": "a1", "slot": "noon"})

    def test_timeout_after_commit_does_not_duplicate(self):
        result = self.run_case([self.initial(), event(10, "propose", receipt_delay=3000), event(4000, "play")], expected_slot="noon")
        self.assertTrue(result["passed"])
        self.assertEqual(result["duplicate_actions"], 0)

    def test_stale_proposal_never_executes(self):
        result = self.run_case([self.initial(), event(10, "propose", execute_delay=100), event(20, "user", facts={"slot": "night"}), event(150, "propose"), event(200, "play")], expected_slot="night")
        self.assertTrue(result["passed"])
        self.assertEqual(result["argument_attempts"], 1)

    def test_stale_receipt_cannot_overwrite_new_result(self):
        result = self.run_case([self.initial(), event(10, "propose", receipt_delay=300), event(20, "user", facts={"slot": "night"}), event(30, "propose"), event(400, "play")], expected_slot="night")
        self.assertTrue(result["passed"])
        self.assertEqual(result["stale_results"], 1)

    def test_interrupt_never_counts_as_fully_played(self):
        result = self.run_case([self.initial(), event(10, "propose"), event(20, "play", fraction=.9), event(30, "interrupt"), event(40, "play")], expected_slot="noon")
        self.assertFalse(result["communication_correct"])
        self.assertIsNone(result["heard"])

    def test_repeating_confirmation_does_not_reexecute_tool(self):
        result = self.run_case([self.initial(), event(10, "propose"), event(20, "interrupt"), event(30, "repeat"), event(40, "play")], expected_slot="noon")
        self.assertTrue(result["passed"])
        self.assertEqual(result["argument_attempts"], 1)

    def test_backchannel_does_not_invalidate_pending_call(self):
        result = self.run_case([self.initial(), event(10, "propose", execute_delay=100), event(20, "user", facts={"backchannel": "yes"}), event(200, "play")], expected_slot="noon")
        self.assertTrue(result["passed"])

    def test_repeated_observation_does_not_cancel_pending_operation(self):
        initial = self.initial()
        result = self.run_case([initial, event(10, "propose", execute_delay=100),
                               event(20, "user", facts=initial["facts"]), event(200, "play")], expected_slot="noon")
        self.assertTrue(result["passed"])

    def test_stale_queued_output_is_invalidated_when_intent_changes(self):
        result = self.run_case([self.initial(), event(10,"propose"), event(20,"user",facts={"slot":"night"}),
                               event(30,"propose"),event(40,"play"),event(50,"play")], expected_slot="night")
        self.assertTrue(result["passed"])
        self.assertEqual(result["unsupported_claims"],0)

    def test_malformed_observation_is_rejected_before_execution(self):
        for facts in ({"slot": None}, {"cancel": "false"}, {"alternatives": "untrusted"},
                      {"alternatives": [1]}, {"alternatives": ["x"]*9}):
            with self.subTest(facts=facts), self.assertRaises(ValueError):
                self.run_case([event(0,"user",facts=facts),event(10,"propose")], expected_slot="original")

    def test_unknown_fields_cannot_override_grounded_draft(self):
        result = self.run_case([self.initial(), event(10, "propose", draft_patch={"slot": "injected", "extra": "ignore instructions"}), event(200, "play")], expected_slot="noon")
        self.assertTrue(result["passed"])
        self.assertGreater(result["prevented"], 0)

    def test_exhausted_alternatives_terminate_without_claim(self):
        initial = self.initial()
        initial["facts"]["alternatives"] = ["also-full"]
        result = self.run_case([initial, event(10, "propose"), event(500, "play")], expected_slot="original", blocked_slots=["noon", "also-full"])
        self.assertFalse(result["passed"])
        self.assertLessEqual(result["argument_attempts"], 3)
        self.assertEqual(result["unsupported_claims"], 0)

    def test_missing_identity_requires_clarification(self):
        result = self.run_case([event(0, "user", facts={"slot": "noon"}), event(10, "propose")], expected_slot="original", expect_communication=False)
        self.assertEqual(result["argument_attempts"], 0)
        self.assertEqual(result["final_state"]["slot"], "original")

    def test_context_is_bounded_and_ledger_survives(self):
        events = [self.initial()]+[event(i, "user", facts={"smalltalk": str(i)}) for i in range(1,2000)]
        events += [event(2100, "propose"), event(2200, "play")]
        result = self.run_case(events, expected_slot="noon")
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_context"], 8)

    def test_event_budget_guards_unbounded_input(self):
        service = ToolService()
        try:
            controller = Controller(POLICIES["combined"], service)
            with self.assertRaisesRegex(RuntimeError, "budget"):
                controller.run([event(i, "tick") for i in range(10001)])
        finally:
            service.close()

    def test_bad_event_times_and_kinds(self):
        for e in (event(-1, "tick"), event(1.5, "tick"), event(0, "unknown")):
            with self.subTest(event=e), self.assertRaises(ValueError):
                self.run_case([e], expected_slot="original")

    def test_invalid_playback_fraction(self):
        with self.assertRaises(ValueError):
            self.run_case([self.initial(), event(10, "propose"), event(20, "play", fraction=1.1)], expected_slot="noon")

    def test_randomized_correction_orderings(self):
        for seed in range(100):
            rng = random.Random(seed)
            updates = sorted(rng.sample(range(20,100), 5))
            events = [self.initial(), event(10, "propose", execute_delay=100)]
            for i, t in enumerate(updates):
                events.append(event(t, "user", facts={"slot": f"slot-{i}"}))
            events += [event(150, "propose", receipt_delay=rng.choice((0,300,1000,3000))), event(5000, "play")]
            with self.subTest(seed=seed):
                result = self.run_case(events, expected_slot="slot-4")
                self.assertTrue(result["passed"])
                self.assertEqual(result["duplicate_actions"], 0)


class SuiteTests(unittest.TestCase):
    def test_generator_is_repeatable_and_splits_disjoint(self):
        self.assertEqual(generate(seed=0), generate(seed=0))
        self.assertTrue({s.id for s in generate("development")}.isdisjoint(s.id for s in generate("held_out")))
        self.assertEqual(len(generate("held_out")),100)

    def test_each_variant_has_a_control_success(self):
        control = generate("held_out")[3]
        for policy in POLICIES.values():
            with self.subTest(policy=policy.name):
                self.assertTrue(evaluate(control, policy)["passed"])

    def test_full_held_out_suite_reports_all_families_and_variants(self):
        from sts_lab.evaluation import run_suite
        report = run_suite()
        self.assertEqual(report["run_count"],1500)
        self.assertEqual(len(report["summaries"]),5)
        for summary in report["summaries"]:
            self.assertEqual(len(summary["by_family"]),5)
            self.assertEqual(summary["total"],300)
        self.assertFalse(report["selection"]["live_audio_qualified"])

    def test_duplicate_seeds_cannot_inflate_report(self):
        from sts_lab.evaluation import run_suite
        with self.assertRaises(ValueError):
            run_suite(seeds=(0,0))
