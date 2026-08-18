"""Tests for the ADK integration.

ADK is not a dependency here: these fakes mimic the shape of the objects ADK
passes to callbacks, which is also the contract the integration is written
against. Run the real thing with examples/adk_agent.py.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardianops import policy  # noqa: E402
from guardianops.adk import (  # noqa: E402
    GuardianOpsGuard,
    scope_drift,
    LexicalContextScorer,
    _extract_objective,
)


class FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeContext:
    """Stands in for ADK's CallbackContext / ToolContext."""

    def __init__(self, invocation_id: str = "inv-1", agent_name: str = "deployer") -> None:
        self.invocation_id = invocation_id
        self.agent_name = agent_name


class FakePart:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeContent:
    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.parts = [FakePart(text)]


class FakeLlmRequest:
    def __init__(self, contents: list) -> None:
        self.contents = contents
        self.config = None


class DenyingApprover:
    async def request(self, panel, timeout):
        return "block"


class ApprovingApprover:
    def __init__(self, answer: str = "allow") -> None:
        self.answer = answer
        self.calls = 0

    async def request(self, panel, timeout):
        self.calls += 1
        return self.answer


def make_guard(tmp: Path, approver=None, **overrides) -> GuardianOpsGuard:
    cfg = policy.Config(
        mode=policy.ENFORCE,
        allow=["read_file", "deploy", "transfer_to_agent"],
        deny=["exfiltrate"],
        tiers={policy.READ: ["read_file"], policy.DESTRUCTIVE: ["deploy"]},
    )
    cfg.audit_path = str(tmp / "audit.jsonl")
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return GuardianOpsGuard(
        cfg=cfg,
        approver=approver or DenyingApprover(),
        baseline_path=str(tmp / "baseline.json"),
        app_name="test-app",
    )


class TestContextScoring(unittest.TestCase):
    def test_on_objective_call_scores_low(self):
        scorer = LexicalContextScorer()
        score = scorer.score(
            "Update the request timeout in mcp.json",
            "read_file",
            {"path": "mcp.json"},
        )
        self.assertLess(score, 0.7)

    def test_unrelated_call_scores_high(self):
        scorer = LexicalContextScorer()
        score = scorer.score(
            "Update the request timeout in mcp.json",
            "query_customers",
            {"table": "customer_payment_records"},
        )
        self.assertGreater(score, 0.8)

    def test_identifiers_are_split_and_lightly_stemmed(self):
        """list_servers should match "server" in the objective."""
        scorer = LexicalContextScorer()
        objective = "Set the request timeout for the docs server"
        on_topic = scorer.score(objective, "list_servers", {})
        off_topic = scorer.score(objective, "drop_table", {})
        self.assertLess(on_topic, off_topic)

    def test_no_objective_invents_no_signal(self):
        self.assertEqual(LexicalContextScorer().score("", "anything", {"a": 1}), 0.0)

    def test_objective_is_the_latest_user_turn(self):
        """ADK sends the whole session, so the first turn is the conversation's
        opener, not this invocation's request. Judging turn five against turn
        one is how one off-scope opener refuses the rest of the session."""
        request = FakeLlmRequest([
            FakeContent("user", "what is the capital of France?"),
            FakeContent("model", "..."),
            FakeContent("user", "what time is it in Kolkata?"),
        ])
        self.assertEqual(_extract_objective(request), "what time is it in Kolkata?")

    def test_a_short_follow_on_inherits_the_previous_turn(self):
        """'and hurry' has no subject of its own; it should not become one."""
        request = FakeLlmRequest([
            FakeContent("model", "I will help with that."),
            FakeContent("user", "Register the docs MCP server"),
            FakeContent("user", "and hurry"),
        ])
        self.assertEqual(
            _extract_objective(request), "Register the docs MCP server and hurry"
        )

    def test_missing_contents_degrade_quietly(self):
        self.assertEqual(_extract_objective(object()), "")


def _adk_installed() -> bool:
    try:
        import google.adk.plugins.base_plugin  # noqa: F401
    except ImportError:
        return False
    return True


class TestScope(unittest.IsolatedAsyncioTestCase):
    """Scope asks whether the *request* belongs to this agent at all.

    Context drift cannot answer that: it scores a tool call against the
    request, and an off-topic request is perfectly consistent with itself.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def scoped(self, threshold=0.9, mode=policy.ENFORCE):
        guard = make_guard(self.tmp, mode=mode)
        guard.cfg.scope = policy.ScopeConfig(
            terms=["mcp", "server", "tool", "policy"],
            threshold=threshold,
            message="out of scope",
        )
        return guard

    def test_drift_is_measured_over_the_request_not_the_terms(self):
        """A short off-topic question must score high against a long term list."""
        terms = ["mcp", "server", "tool", "policy", "audit", "ledger", "drift"]
        self.assertLess(scope_drift("list the mcp servers", terms), 0.5)
        self.assertEqual(scope_drift("write me a poem about the sea", terms), 1.0)

    def test_no_terms_means_no_check(self):
        """An agent with no declared purpose has nothing to be out of scope of."""
        self.assertEqual(scope_drift("anything at all", []), 0.0)
        self.assertFalse(policy.ScopeConfig().enabled)

    async def test_threshold_none_scores_without_enforcing(self):
        guard = self.scoped(threshold=None)
        result = await guard.before_model_callback(
            FakeContext(), FakeLlmRequest([FakeContent("user", "bake a cake")])
        )
        self.assertIsNone(result)
        self.assertIn('"scope_drift":1.0', Path(guard.cfg.audit_path).read_text())

    async def test_in_scope_request_passes(self):
        guard = self.scoped()
        result = await guard.before_model_callback(
            FakeContext(), FakeLlmRequest([FakeContent("user", "list the mcp servers")])
        )
        self.assertIsNone(result)

    async def test_shadow_mode_scores_but_never_turns_away(self):
        guard = self.scoped(mode=policy.SHADOW)
        result = await guard.before_model_callback(
            FakeContext(), FakeLlmRequest([FakeContent("user", "bake a cake")])
        )
        self.assertIsNone(result)
        self.assertIn('"out_of_scope":true', Path(guard.cfg.audit_path).read_text())

    @unittest.skipUnless(_adk_installed(), "refusal object needs ADK")
    async def test_an_off_scope_opener_does_not_poison_the_session(self):
        """The regression: ADK sends the whole conversation, so turn two must be
        judged on turn two."""
        guard = self.scoped()
        turn_one = await guard.before_model_callback(
            FakeContext(invocation_id="t1"),
            FakeLlmRequest([FakeContent("user", "bake a cake")]),
        )
        self.assertIsNotNone(turn_one)

        turn_two = await guard.before_model_callback(
            FakeContext(invocation_id="t2"),
            FakeLlmRequest([
                FakeContent("user", "bake a cake"),
                FakeContent("model", "out of scope"),
                FakeContent("user", "which mcp server tools are allowed?"),
            ]),
        )
        self.assertIsNone(turn_two)

    @unittest.skipUnless(_adk_installed(), "refusal object needs ADK")
    async def test_deny_pattern_refuses_before_inference(self):
        """A categorical exclusion the topic score is too blunt to express."""
        guard = self.scoped()
        guard.cfg.scope.deny_patterns = [r"\b(berlin|london)\b"]
        refused = await guard.before_model_callback(
            FakeContext(invocation_id="d1"),
            FakeLlmRequest([FakeContent("user", "what time is it in Berlin?")]),
        )
        self.assertIsNotNone(refused)
        self.assertIn('"denied_pattern"', Path(guard.cfg.audit_path).read_text())

    async def test_deny_pattern_does_not_catch_a_paraphrase(self):
        """Stated, so nobody mistakes a word filter for comprehension."""
        guard = self.scoped()
        guard.cfg.scope.deny_patterns = [r"\bberlin\b"]
        allowed = await guard.before_model_callback(
            FakeContext(invocation_id="d2"),
            FakeLlmRequest([
                FakeContent("user", "what is the mcp server tool time in Germany's capital?")
            ]),
        )
        self.assertIsNone(allowed)

    def test_a_broken_pattern_does_not_break_the_agent(self):
        scope = policy.ScopeConfig(deny_patterns=["([unclosed"], threshold=0.9)
        self.assertIsNone(scope.denied("anything"))

    async def test_out_of_scope_run_blocks_tools_too(self):
        """A model that ignores the refusal still gets nothing."""
        guard = self.scoped()
        ctx = FakeContext()
        await guard.before_model_callback(
            ctx, FakeLlmRequest([FakeContent("user", "bake a cake")])
        )
        blocked = await guard.before_tool_callback(FakeTool("read_file"), {"path": "/x"}, ctx)
        self.assertIsNotNone(blocked)
        self.assertIn("scope", blocked["reason"])


class TestGuardDecisions(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    async def test_unentitled_tool_returns_a_blocking_result(self):
        guard = make_guard(self.tmp)
        result = await guard.before_tool_callback(
            FakeTool("attach_role_policy"), {"role": "admin"}, FakeContext()
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["error"], "blocked_by_guardianops")
        self.assertIn("privilege drift", result["reason"])

    async def test_allowed_tool_returns_none_so_adk_proceeds(self):
        guard = make_guard(self.tmp)
        result = await guard.before_tool_callback(
            FakeTool("read_file"), {"path": "/x"}, FakeContext()
        )
        self.assertIsNone(result)

    async def test_shadow_mode_never_blocks(self):
        guard = make_guard(self.tmp, mode=policy.SHADOW)
        result = await guard.before_tool_callback(
            FakeTool("attach_role_policy"), {}, FakeContext()
        )
        self.assertIsNone(result)

    async def test_destructive_tier_is_held_and_denial_blocks(self):
        approver = ApprovingApprover(answer="block")
        guard = make_guard(self.tmp, approver=approver)
        result = await guard.before_tool_callback(FakeTool("deploy"), {}, FakeContext())
        self.assertEqual(approver.calls, 1)
        self.assertIsNotNone(result)

    async def test_approval_lets_the_call_through(self):
        guard = make_guard(self.tmp, approver=ApprovingApprover("allow"))
        result = await guard.before_tool_callback(FakeTool("deploy"), {}, FakeContext())
        self.assertIsNone(result)

    async def test_whitelist_is_not_asked_twice(self):
        approver = ApprovingApprover("whitelist")
        guard = make_guard(self.tmp, approver=approver)
        ctx = FakeContext()
        await guard.before_tool_callback(FakeTool("deploy"), {}, ctx)
        await guard.before_tool_callback(FakeTool("deploy"), {}, ctx)
        self.assertEqual(approver.calls, 1)

    async def test_argument_constraints_apply_to_adk_tools(self):
        guard = make_guard(self.tmp)
        guard.cfg.constraints = {
            "read_file": {"path": policy.ArgumentConstraint.parse({"prefix": ["/workspace/"]})}
        }
        allowed = await guard.before_tool_callback(
            FakeTool("read_file"), {"path": "/workspace/a"}, FakeContext()
        )
        blocked = await guard.before_tool_callback(
            FakeTool("read_file"), {"path": "/etc/shadow"}, FakeContext()
        )
        self.assertIsNone(allowed)
        self.assertIsNotNone(blocked)
        self.assertIn("argument constraint", blocked["reason"])

    async def test_on_decision_observer_sees_every_decision(self):
        seen = []
        guard = make_guard(self.tmp)
        guard.on_decision = seen.append
        await guard.before_tool_callback(FakeTool("read_file"), {"path": "/x"}, FakeContext())
        await guard.before_tool_callback(FakeTool("attach_role_policy"), {}, FakeContext())
        self.assertEqual([r["tool"] for r in seen], ["read_file", "attach_role_policy"])
        self.assertEqual([r["outcome"] for r in seen], ["allow", "block"])

    async def test_a_raising_observer_does_not_break_the_call(self):
        """An observer is a mirror, not a control. Governance already happened."""
        def boom(_record):
            raise RuntimeError("dashboard is down")

        guard = make_guard(self.tmp)
        guard.on_decision = boom
        result = await guard.before_tool_callback(
            FakeTool("read_file"), {"path": "/x"}, FakeContext()
        )
        self.assertIsNone(result)

    async def test_credentials_in_tool_args_are_redacted_in_the_ledger(self):
        guard = make_guard(self.tmp)
        await guard.before_tool_callback(
            FakeTool("read_file"), {"path": "/x", "api_key": "sk-live"}, FakeContext()
        )
        text = Path(guard.cfg.audit_path).read_text()
        self.assertNotIn("sk-live", text)
        self.assertIn("***redacted***", text)


class TestRunTracking(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    async def test_one_invocation_is_one_run_across_agents(self):
        """A multi-agent handoff must trace as a single run, not two."""
        guard = make_guard(self.tmp)
        planner = FakeContext(invocation_id="inv-9", agent_name="planner")
        executor = FakeContext(invocation_id="inv-9", agent_name="executor")
        await guard.before_tool_callback(FakeTool("read_file"), {}, planner)
        await guard.before_tool_callback(FakeTool("read_file"), {}, executor)
        self.assertEqual(len(guard._runs), 1)
        self.assertEqual(guard._runs["inv-9"].calls_total, 2)

    async def test_separate_invocations_do_not_share_counters(self):
        guard = make_guard(self.tmp)
        await guard.before_tool_callback(FakeTool("read_file"), {}, FakeContext("inv-a"))
        await guard.before_tool_callback(FakeTool("read_file"), {}, FakeContext("inv-b"))
        self.assertEqual(len(guard._runs), 2)

    async def test_completed_runs_are_released(self):
        """A long-lived ADK service must not accumulate finished runs."""
        guard = make_guard(self.tmp)
        for i in range(200):
            ctx = FakeContext(invocation_id=f"inv-{i}")
            await guard.before_agent_callback(ctx)
            await guard.before_tool_callback(FakeTool("read_file"), {}, ctx)
            await guard.after_agent_callback(ctx)
        self.assertEqual(len(guard._runs), 0)

    async def test_a_multi_agent_run_is_held_until_the_last_agent_ends(self):
        guard = make_guard(self.tmp)
        planner = FakeContext("inv-1", "planner")
        executor = FakeContext("inv-1", "executor")
        await guard.before_agent_callback(planner)
        await guard.before_agent_callback(executor)
        await guard.after_agent_callback(executor)
        self.assertEqual(len(guard._runs), 1)  # planner is still running
        await guard.after_agent_callback(planner)
        self.assertEqual(len(guard._runs), 0)

    async def test_abandoned_runs_are_evicted_by_ttl(self):
        """Agents crash mid-turn; refcounting alone would leak those forever."""
        guard = make_guard(self.tmp)
        guard.run_ttl_seconds = 0.0
        await guard.before_tool_callback(FakeTool("read_file"), {}, FakeContext("abandoned"))
        await guard.before_tool_callback(FakeTool("read_file"), {}, FakeContext("fresh"))
        self.assertNotIn("abandoned", guard._runs)

    async def test_run_count_is_capped(self):
        guard = make_guard(self.tmp)
        guard.max_runs = 10
        for i in range(50):
            await guard.before_tool_callback(FakeTool("read_file"), {}, FakeContext(f"inv-{i}"))
        self.assertLessEqual(len(guard._runs), 11)

    async def test_delegation_counts_toward_autonomy(self):
        guard = make_guard(self.tmp)
        ctx = FakeContext()
        await guard.before_tool_callback(
            FakeTool("transfer_to_agent"), {"agent_name": "executor"}, ctx
        )
        self.assertEqual(guard._runs[ctx.invocation_id].transfers, 1)

    async def test_autonomy_drift_escalates_after_threshold(self):
        approver = ApprovingApprover("allow")
        guard = make_guard(self.tmp, approver=approver)
        guard.cfg.approval.autonomy_threshold = 3
        ctx = FakeContext()
        for _ in range(4):
            await guard.before_tool_callback(FakeTool("read_file"), {}, ctx)
        self.assertGreaterEqual(approver.calls, 1)

    async def test_approval_resets_the_autonomy_counter(self):
        guard = make_guard(self.tmp, approver=ApprovingApprover("allow"))
        guard.cfg.approval.autonomy_threshold = 3
        ctx = FakeContext()
        for _ in range(4):
            await guard.before_tool_callback(FakeTool("read_file"), {}, ctx)
        self.assertEqual(guard._runs[ctx.invocation_id].calls_since_approval, 0)

    async def test_baseline_is_scoped_per_agent(self):
        guard = make_guard(self.tmp)
        await guard.before_tool_callback(
            FakeTool("read_file"), {}, FakeContext(agent_name="planner")
        )
        self.assertFalse(guard.baseline.is_novel("test-app/planner", "read_file"))
        # The same tool called by a different agent is still a first for that agent.
        self.assertTrue(guard.baseline.is_novel("test-app/executor", "read_file"))


class TestContextDriftEnforcement(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    async def test_context_is_scored_but_not_enforced_by_default(self):
        guard = make_guard(self.tmp)
        ctx = FakeContext()
        await guard.before_model_callback(
            ctx, FakeLlmRequest([FakeContent("user", "Update the mcp.json timeout")])
        )
        result = await guard.before_tool_callback(
            FakeTool("read_file"), {"table": "customer_payment_records"}, ctx
        )
        self.assertIsNone(result)  # scored, allowed
        text = Path(guard.cfg.audit_path).read_text()
        self.assertIn('"context"', text)

    async def test_context_drift_escalates_when_a_threshold_is_set(self):
        approver = ApprovingApprover("block")
        guard = make_guard(self.tmp, approver=approver)
        guard.cfg.approval.context_threshold = 0.8
        ctx = FakeContext()
        await guard.before_model_callback(
            ctx, FakeLlmRequest([FakeContent("user", "Update the mcp.json timeout")])
        )
        result = await guard.before_tool_callback(
            FakeTool("read_file"), {"table": "customer_payment_records"}, ctx
        )
        self.assertIsNotNone(result)
        self.assertIn("context drift", result["reason"])

    async def test_control_plane_calls_are_exempt_from_context_drift(self):
        """Agent handoff carries agent names, not subject matter. Scoring it
        against the objective flags every multi-agent run as drifting."""
        guard = make_guard(self.tmp, approver=ApprovingApprover("block"))
        guard.cfg.approval.context_threshold = 0.8
        guard.cfg.tiers["control"] = ["transfer_to_agent"]
        ctx = FakeContext()
        await guard.before_model_callback(
            ctx, FakeLlmRequest([FakeContent("user", "Update the mcp.json timeout")])
        )
        result = await guard.before_tool_callback(
            FakeTool("transfer_to_agent"), {"agent_name": "executor"}, ctx
        )
        self.assertIsNone(result)

    async def test_exemption_does_not_extend_to_other_tiers(self):
        guard = make_guard(self.tmp, approver=ApprovingApprover("block"))
        guard.cfg.approval.context_threshold = 0.8
        guard.cfg.tiers["control"] = ["transfer_to_agent"]
        ctx = FakeContext()
        await guard.before_model_callback(
            ctx, FakeLlmRequest([FakeContent("user", "Update the mcp.json timeout")])
        )
        result = await guard.before_tool_callback(
            FakeTool("read_file"), {"table": "customer_payment_records"}, ctx
        )
        self.assertIsNotNone(result)

    async def test_objective_is_captured_once_per_run(self):
        guard = make_guard(self.tmp)
        ctx = FakeContext()
        await guard.before_model_callback(ctx, FakeLlmRequest([FakeContent("user", "First goal")]))
        await guard.before_model_callback(ctx, FakeLlmRequest([FakeContent("user", "Later turn")]))
        self.assertEqual(guard._runs[ctx.invocation_id].objective, "First goal")


class TestInstall(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_install_walks_sub_agents(self):
        class FakeAgent:
            def __init__(self, name, subs=()):
                self.name = name
                self.sub_agents = list(subs)
                self.before_tool_callback = None
                self.after_tool_callback = None
                self.before_model_callback = None
                self.before_agent_callback = None
                self.after_agent_callback = None

        child = FakeAgent("executor")
        root = FakeAgent("planner", [child])
        make_guard(self.tmp).install(root)
        self.assertIsNotNone(root.before_tool_callback)
        self.assertIsNotNone(child.before_tool_callback)

    def test_install_does_not_clobber_existing_callbacks(self):
        class FakeAgent:
            def __init__(self):
                self.name = "a"
                self.sub_agents = []
                self.before_tool_callback = "mine"

        agent = FakeAgent()
        make_guard(self.tmp).install(agent)
        self.assertEqual(agent.before_tool_callback, "mine")


@unittest.skipUnless(_adk_installed(), "google-adk not installed")
class TestPlugin(unittest.IsolatedAsyncioTestCase):
    """The plugin path, checked against a real ADK BasePlugin.

    Skipped when ADK is absent, which is why the rest of this file uses fakes.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_plugin_wires_every_hook_the_guard_needs(self):
        """The agent hooks are load-bearing: after_agent_callback is where the
        baseline is saved. A plugin missing them silently never learns."""
        plugin = make_guard(self.tmp).as_plugin()
        from google.adk.plugins.base_plugin import BasePlugin

        for hook in (
            "before_agent_callback",
            "after_agent_callback",
            "before_model_callback",
            "before_tool_callback",
            "after_tool_callback",
        ):
            self.assertIsNot(
                getattr(type(plugin), hook),
                getattr(BasePlugin, hook),
                f"{hook} left as the ADK no-op",
            )

    async def test_plugin_enforces_and_persists_the_baseline(self):
        guard = make_guard(self.tmp)
        plugin = guard.as_plugin()
        ctx = FakeContext()

        await plugin.before_agent_callback(agent=None, callback_context=ctx)
        blocked = await plugin.before_tool_callback(
            tool=FakeTool("attach_role_policy"), tool_args={}, tool_context=ctx
        )
        self.assertIsNotNone(blocked)
        allowed = await plugin.before_tool_callback(
            tool=FakeTool("read_file"), tool_args={"path": "/x"}, tool_context=ctx
        )
        self.assertIsNone(allowed)
        await plugin.after_tool_callback(
            tool=FakeTool("read_file"), tool_args={"path": "/x"},
            tool_context=ctx, result={"ok": True},
        )
        await plugin.after_agent_callback(agent=None, callback_context=ctx)

        self.assertTrue(Path(guard.baseline.path).exists())


if __name__ == "__main__":
    unittest.main()
