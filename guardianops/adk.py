"""Google ADK integration.

ADK exposes callbacks around the agent loop -- ``before_agent``, ``before_model``,
``before_tool``, ``after_tool`` -- and a ``before_tool_callback`` that returns a
dict short-circuits the tool and hands that dict back to the model as the tool
result. That is exactly the shape of an inline governance decision, so the whole
policy engine drops straight in.

What this buys over the MCP proxy:

  * **Every tool, not just MCP ones.** FunctionTool, AgentTool, OpenAPI toolsets
    and MCPToolset tools all pass through the same callback.
  * **Context drift becomes computable.** ``before_model_callback`` sees the
    LlmRequest, so the agent's stated objective is observable in-process. The
    MCP boundary never sees it.
  * **Real run identity.** invocation_id, agent name and session id come from
    ADK, so a multi-agent run traces as one thing instead of N disconnected
    sessions.
  * **Delegation is visible.** ``transfer_to_agent`` is an ordinary tool call
    here, so sub-agent handoff is governable and counts toward autonomy drift.

What it does *not* buy: this runs inside the agent process, so it is trivially
bypassable by anyone editing the agent. It is a fidelity and coverage layer, not
an enforcement boundary. Keep the MCP proxy in front of MCP servers and lock
egress at the network -- see the README.

Nothing here imports ADK at module load, so the module is importable (and
testable) without ADK installed. Every ADK object is read defensively with
``getattr`` so a version bump that renames a field degrades to "unknown" rather
than crashing the agent.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from . import hitl, policy
from .audit import AuditLog
from .baseline import InvocationBaseline
from .proxy import log

# ---------------------------------------------------------------- context drift

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have i if in into is it its of on or
    that the then there these this to was will with you your please can could
    should would do does did make sure need want use using""".split()
)


def _tokens(text: str) -> set[str]:
    """Split into comparable terms.

    Identifiers are split on underscores so ``list_servers`` can match "server"
    in an objective, and a trailing plural is stripped -- crude stemming, but the
    alternative is scoring ``servers`` and ``server`` as unrelated words, which
    is worse than crude.
    """
    out: set[str] = set()
    for token in _TOKEN.findall(text.lower()):
        for part in token.split("_"):
            if len(part) <= 2 or part in _STOPWORDS:
                continue
            if len(part) > 4 and part.endswith("s") and not part.endswith("ss"):
                part = part[:-1]
            out.add(part)
    return out


def scope_drift(objective: str, terms: list[str]) -> float:
    """How much of the request has nothing to do with the agent's purpose.

    0.0 means every meaningful word in the request relates to the declared
    terms, 1.0 means none of them do. Deliberately measured over the request's
    own vocabulary rather than the terms': a short off-topic question should
    score high even against a long term list.
    """
    asked = _tokens(objective)
    if not asked or not terms:
        return 0.0
    in_scope = _tokens(" ".join(terms))
    return round(1.0 - len(asked & in_scope) / len(asked), 3)


def _refusal(message: str) -> Any:
    """An LlmResponse that replaces the model's answer.

    ADK is imported here rather than at module load so the guard stays usable
    (and testable) without it. If the import fails we return None: failing to
    build a refusal must not crash the agent, and the tool-level controls still
    apply to whatever the model does next.
    """
    try:
        from google.adk.models.llm_response import LlmResponse  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError:  # pragma: no cover - only without ADK installed
        log("cannot build a refusal without ADK; letting the request through")
        return None
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=message)])
    )


class ContextScorer(Protocol):
    """Scores how far a tool call sits from the run's stated objective.

    0.0 means squarely on objective, 1.0 means unrelated.
    """

    def score(self, objective: str, tool: str, args: dict[str, Any]) -> float: ...


class LexicalContextScorer:
    """Token-overlap stand-in for embedding-based context drift.

    This is deliberately crude: it measures vocabulary overlap between the
    objective and the call, which catches an agent that wandered onto an
    unrelated subject and misses one that stayed on topic while doing something
    dangerous. It exists so the signal is wired end to end and can be tuned
    against real traffic; swap in an encoder by passing any ContextScorer to
    GuardianOpsGuard. Because it is approximate, ``approval.contextThreshold``
    defaults to off -- the score is recorded, not acted on.
    """

    def score(self, objective: str, tool: str, args: dict[str, Any]) -> float:
        objective_tokens = _tokens(objective)
        if not objective_tokens:
            return 0.0  # nothing to compare against; do not invent a signal
        call_tokens = _tokens(f"{tool} {args}")
        if not call_tokens:
            return 0.0
        overlap = len(objective_tokens & call_tokens)
        return round(1.0 - overlap / len(call_tokens), 3)


# ------------------------------------------------------------------ run tracking


@dataclass
class _Run:
    run_id: str
    agent: str = "unknown"
    objective: str = ""
    calls_total: int = 0
    calls_since_approval: int = 0
    blocked: int = 0
    escalated: int = 0
    transfers: int = 0
    session_allow: set[str] = field(default_factory=set)
    tools_seen: set[str] = field(default_factory=set)
    # Set when the request was turned away as off scope, so a model that
    # somehow still reaches for a tool gets nothing.
    out_of_scope: bool = False
    # How many agents in this invocation have started and not yet finished. A
    # multi-agent run shares one _Run, so it is released when the last one ends.
    active_agents: int = 0
    last_seen: float = field(default_factory=time.monotonic)


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Read the first attribute that exists. ADK field names move between versions."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _extract_objective(llm_request: Any) -> str:
    """Pull the request this invocation is actually serving out of an LlmRequest.

    The *latest* user turn, not the first. ``contents`` carries the whole
    session once a conversation has more than one turn, so reading the first
    user turn pins the objective to whatever the conversation opened with --
    and every later message gets judged against a question nobody is asking any
    more. In a chat that is not a subtle inaccuracy: one off-scope opener would
    turn away the rest of the session.

    A short follow-on ("and hurry", "do it") carries no subject of its own, so
    it inherits the previous user turn rather than becoming the objective.
    Falls back to the system instruction, which says what the agent always is
    when nothing says what this run is for.
    """
    contents = _attr(llm_request, "contents", default=[]) or []
    turns: list[str] = []
    for content in contents:
        if _attr(content, "role") != "user":
            continue
        parts = _attr(content, "parts", default=[]) or []
        text = " ".join(str(_attr(p, "text", default="")) for p in parts).strip()
        if text:
            turns.append(text)

    if turns:
        latest = turns[-1]
        if len(_tokens(latest)) < 3 and len(turns) > 1:
            return f"{turns[-2]} {latest}"
        return latest

    config = _attr(llm_request, "config")
    instruction = _attr(config, "system_instruction", "systemInstruction", default="")
    return str(instruction) if instruction else ""


# ------------------------------------------------------------------- the guard


class GuardianOpsGuard:
    """Governance callbacks for ADK agents.

    Attach to one agent::

        guard = GuardianOpsGuard(config_path="config.adk.json")
        agent = LlmAgent(..., **guard.callbacks())

    or to every agent in a Runner, if your ADK version has the plugin API::

        runner = Runner(..., plugins=[guard.as_plugin()])
    """

    def __init__(
        self,
        config_path: str | None = None,
        *,
        cfg: policy.Config | None = None,
        approver: Any | None = None,
        baseline_path: str = "",
        context_scorer: ContextScorer | None = None,
        app_name: str = "",
        max_runs: int = 2048,
        run_ttl_seconds: float = 3600.0,
        on_decision: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        # Called with each tool.call record after the decision is made. Purely
        # an observer -- it cannot change the outcome. See _emit.
        self.on_decision = on_decision
        self.max_runs = max_runs
        self.run_ttl_seconds = run_ttl_seconds
        self.cfg = cfg or policy.Config.load(config_path)
        self.approver = approver or hitl.TtyApprover()
        # Everything not passed comes from the policy, so declaring an agent is
        # writing one file rather than writing one file and remembering four
        # arguments that have to agree with it.
        self.baseline = InvocationBaseline(baseline_path or self.cfg.baseline_path)
        self.context_scorer = context_scorer or LexicalContextScorer()
        self.app_name = app_name or self.cfg.name or "adk"
        self.audit = AuditLog(
            self.cfg.audit_path,
            run_id=f"adk-{uuid.uuid4().hex[:8]}",
            sync_mode=self.cfg.audit_sync,
            key=self.cfg.audit_key(),
        )
        self._runs: dict[str, _Run] = {}

    @classmethod
    def from_policy(cls, path: str, **overrides: Any) -> "GuardianOpsGuard":
        """Build a guard from a policy file and nothing else.

            guard = GuardianOpsGuard.from_policy("policies/agents/chat.json")
            agent = LlmAgent(..., **guard.callbacks())

        Name, mode, ledger, pins and baseline all come from the file, and its
        relative paths resolve next to it. Adding an agent is adding a file.
        """
        return cls(config_path=path, **overrides)

    # -- observers ----------------------------------------------------------

    def _emit(self, record: dict[str, Any]) -> None:
        """Hand a decision to ``on_decision``, if anyone set one.

        The ledger is the durable record; this is for surfacing a decision
        somewhere a human is already looking -- a UI, a dashboard, a log
        shipper. An observer that raises must not take the agent down with it,
        so failures are logged and swallowed: governance already happened by
        the time this runs.
        """
        if self.on_decision is None:
            return
        try:
            self.on_decision(record)
        except Exception as exc:  # noqa: BLE001 - an observer is not load-bearing
            log(f"on_decision observer raised, ignoring: {exc!r}")

    # -- identity -----------------------------------------------------------

    def _run_for(self, ctx: Any) -> _Run:
        """One _Run per ADK invocation, shared across every agent in the run."""
        run_id = str(
            _attr(ctx, "invocation_id", "invocationId", default=None) or self.audit.run_id
        )
        run = self._runs.get(run_id)
        if run is None:
            self._evict_stale()
            run = _Run(run_id=run_id)
            self._runs[run_id] = run
        run.last_seen = time.monotonic()
        agent = _attr(ctx, "agent_name", "agentName")
        if agent:
            run.agent = str(agent)
        return run

    def _release(self, run: _Run) -> None:
        """Drop a finished run so a long-lived service does not accumulate them."""
        self._runs.pop(run.run_id, None)

    def _evict_stale(self) -> None:
        """Backstop for runs that never see a clean end.

        Agents crash, get cancelled mid-turn, or run on an ADK version that does
        not fire after_agent. Refcounting alone would leak those forever, so age
        them out and cap the total.
        """
        now = time.monotonic()
        for run_id, run in list(self._runs.items()):
            if now - run.last_seen > self.run_ttl_seconds:
                del self._runs[run_id]
        overflow = len(self._runs) - self.max_runs
        if overflow > 0:
            oldest = sorted(self._runs.items(), key=lambda kv: kv[1].last_seen)
            for run_id, _ in oldest[:overflow]:
                del self._runs[run_id]
            log(f"evicted {overflow} run(s) over the {self.max_runs} cap")

    def _baseline_key(self, run: _Run) -> str:
        """Baselines are per agent. A prompt change is a different agent -- bump
        ``app_name`` (or the agent name) when the prompt changes, or the new
        behavior gets scored against the old baseline."""
        return f"{self.app_name}/{run.agent}"

    # -- callbacks ----------------------------------------------------------

    async def before_agent_callback(self, callback_context: Any) -> None:
        run = self._run_for(callback_context)
        run.active_agents += 1
        self.audit.record(
            "agent.start",
            adk_run=run.run_id,
            agent=run.agent,
            app=self.app_name,
            mode=self.cfg.mode,
        )
        return None

    async def after_agent_callback(self, callback_context: Any) -> None:
        run = self._run_for(callback_context)
        run.active_agents = max(0, run.active_agents - 1)
        self.audit.record(
            "agent.end", adk_run=run.run_id, agent=run.agent, mode=self.cfg.mode
        )
        if run.active_agents == 0:
            # The last agent in the invocation finished: summarize and let go.
            self.audit.record(
                "run.end",
                adk_run=run.run_id,
                agent=run.agent,
                app=self.app_name,
                mode=self.cfg.mode,
                calls_total=run.calls_total,
                blocked=run.blocked,
                escalated=run.escalated,
                transfers=run.transfers,
                distinct_tools=sorted(run.tools_seen),
            )
            self.baseline.save()
            self._release(run)
        return None

    async def before_model_callback(self, callback_context: Any, llm_request: Any) -> Any:
        """Capture the objective, and turn the request away if it is off scope.

        Returning an LlmResponse here short-circuits the model call, so an
        out-of-scope request costs nothing and reaches nothing. That is the only
        control here that acts before inference rather than after it.
        """
        run = self._run_for(callback_context)
        if run.objective:
            return None

        objective = _extract_objective(llm_request)
        if not objective:
            return None

        run.objective = objective
        scope = self.cfg.scope
        drift = scope_drift(objective, scope.terms) if scope.enabled else 0.0
        denied = scope.denied(objective) if scope.enabled else None
        out_of_scope = bool(denied) or (
            scope.enabled
            and scope.threshold is not None
            and drift >= scope.threshold
        )
        shadowed = out_of_scope and self.cfg.mode == policy.SHADOW
        turned_away = out_of_scope and not shadowed and scope.on_out_of_scope == policy.BLOCK

        record = self.audit.record(
            "run.objective",
            adk_run=run.run_id,
            agent=run.agent,
            objective=self.cfg.redact({"text": objective})["text"],
            scope_drift=round(drift, 3),
            out_of_scope=out_of_scope,
            denied_pattern=denied,
            shadowed=shadowed,
            outcome=policy.BLOCK if turned_away else policy.ALLOW,
        )
        if out_of_scope:
            verb = "would turn away" if shadowed or not turned_away else "turned away"
            why = f"deny pattern {denied!r}" if denied else f"scope drift {drift:.2f}"
            log(f"{verb} request · {why} · {objective[:60]!r}")
            self._emit(record)
        if turned_away:
            run.out_of_scope = True
            return _refusal(scope.message)
        return None

    async def before_tool_callback(
        self, tool: Any, args: dict[str, Any], tool_context: Any
    ) -> dict[str, Any] | None:
        """The decision point. Returning a dict blocks the call and that dict
        becomes the tool result the model sees."""
        run = self._run_for(tool_context)
        name = str(_attr(tool, "name", default=None) or type(tool).__name__)
        args = args or {}

        # The request was already turned away as off scope. Nothing it asks for
        # is in scope either, so do not re-litigate it per tool.
        if run.out_of_scope:
            log(f"blocked {name} · request was turned away as out of scope")
            return {
                "error": "blocked_by_guardianops",
                "tool": name,
                "reason": "the request is outside this agent's scope",
                "message": (
                    f"GuardianOps blocked this call: {self.cfg.scope.message} "
                    f"Do not answer from memory or substitute a result."
                ),
            }

        if name == "transfer_to_agent":
            run.transfers += 1

        context_score = (
            self.context_scorer.score(run.objective, name, args) if run.objective else 0.0
        )
        key = self._baseline_key(run)

        verdict = policy.evaluate(
            self.cfg,
            name,
            novel=self.baseline.is_novel(key, name),
            pin_changed=False,  # definition pinning lives at the MCP boundary
            calls_since_approval=run.calls_since_approval,
            session_allowed=name in run.session_allow,
            context=context_score,
            arguments=args,
        )

        run.calls_total += 1
        run.calls_since_approval += 1
        run.tools_seen.add(name)

        outcome = verdict.effective
        approval = None
        held = outcome == policy.ESCALATE

        if held:
            run.escalated += 1
            approval = await self._escalate(run, name, args, verdict)
            if approval in (hitl.ALLOW_ONCE, hitl.WHITELIST):
                outcome = policy.ALLOW
                run.calls_since_approval = 0
                if approval == hitl.WHITELIST:
                    run.session_allow.add(name)
            else:
                outcome = policy.BLOCK

        record = self.audit.record(
            "tool.call",
            adk_run=run.run_id,
            agent=run.agent,
            server=key,
            tool=name,
            tier=verdict.tier,
            decision=verdict.decision,
            outcome=outcome,
            held=held,
            shadowed=verdict.shadowed,
            reason=verdict.reason,
            risk=verdict.risk,
            signals=verdict.signals.as_dict(),
            approval=approval,
            prior_invocations=self.baseline.count(key, name),
            arguments=self.cfg.redact(args),
        )
        self._emit(record)

        if outcome == policy.BLOCK:
            run.blocked += 1
            if verdict.shadowed:
                log(f"would have blocked {name} · risk {verdict.risk:.2f} · {verdict.reason}")
                self.baseline.observe(key, name)
                return None
            log(f"blocked {name} · risk {verdict.risk:.2f} · {verdict.reason}")
            return {
                "error": "blocked_by_guardianops",
                "tool": name,
                "reason": verdict.reason,
                "composite_risk": verdict.risk,
                # This text is read by the model, so it is instruction as much
                # as explanation. Without the second sentence a model will
                # happily answer from memory instead -- the call is governed,
                # but the user sees an answer and assumes it was not.
                "message": (
                    f"GuardianOps blocked this call. {verdict.reason}. "
                    f"Do not answer from memory, guess, or otherwise substitute "
                    f"a result for this tool. Tell the user the action was "
                    f"blocked by policy, then request an entitlement change or "
                    f"choose a different approach."
                ),
            }

        if verdict.risk >= 0.4:
            log(f"allowed {name} · risk {verdict.risk:.2f} · {verdict.reason}")
        self.baseline.observe(key, name)
        return None

    async def after_tool_callback(
        self, tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any
    ) -> None:
        run = self._run_for(tool_context)
        name = str(_attr(tool, "name", default=None) or type(tool).__name__)
        is_error = isinstance(tool_response, dict) and (
            "error" in tool_response or tool_response.get("isError") is True
        )
        self.audit.record(
            "tool.result",
            adk_run=run.run_id,
            agent=run.agent,
            tool=name,
            is_error=bool(is_error),
        )
        return None

    async def _escalate(
        self, run: _Run, tool: str, args: dict[str, Any], verdict: policy.Verdict
    ) -> str:
        panel = {
            "run_id": f"{run.run_id} · {run.agent}",
            "server": self.app_name,
            "tool": tool,
            "tier": verdict.tier,
            "reason": verdict.reason,
            "risk": verdict.risk,
            "signals": verdict.signals.as_dict(),
            "arguments": str(self.cfg.redact(args)),
        }
        log(f"HELD {tool} · risk {verdict.risk:.2f} · {verdict.reason}")
        answer = await self.approver.request(panel, self.cfg.approval.timeout_seconds)
        if answer in (hitl.UNAVAILABLE, hitl.TIMEOUT):
            log(f"no approval obtained; applying onTimeout={self.cfg.approval.on_timeout}")
            return (
                hitl.ALLOW_ONCE
                if self.cfg.approval.on_timeout == policy.ALLOW
                else hitl.BLOCK
            )
        return answer

    # -- wiring -------------------------------------------------------------

    def callbacks(self) -> dict[str, Callable]:
        """Keyword arguments for an ADK agent constructor."""
        return {
            "before_agent_callback": self.before_agent_callback,
            "after_agent_callback": self.after_agent_callback,
            "before_model_callback": self.before_model_callback,
            "before_tool_callback": self.before_tool_callback,
            "after_tool_callback": self.after_tool_callback,
        }

    def install(self, agent: Any, *, recursive: bool = True) -> Any:
        """Attach the callbacks to an existing agent, and its sub-agents.

        Only fills callbacks the agent does not already define, so an agent with
        its own hooks keeps them -- but note that means it is *not* governed
        unless you compose the callbacks yourself.
        """
        for name, fn in self.callbacks().items():
            if hasattr(agent, name) and getattr(agent, name, None) is None:
                try:
                    setattr(agent, name, fn)
                except (AttributeError, ValueError) as exc:  # frozen/validated models
                    log(f"could not attach {name} to {_attr(agent, 'name')}: {exc}")
        if recursive:
            for sub in _attr(agent, "sub_agents", "subAgents", default=[]) or []:
                self.install(sub, recursive=True)
        return agent

    def as_plugin(self) -> Any:
        """Build an ADK Plugin so one instance governs every agent in a Runner.

        Requires an ADK version with the plugin API; raises otherwise, in which
        case use ``install()`` or ``callbacks()``.
        """
        try:
            from google.adk.plugins.base_plugin import BasePlugin  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on the installed ADK
            raise RuntimeError(
                "this ADK version has no plugin API; use guard.install(agent) "
                "or **guard.callbacks() instead"
            ) from exc

        guard = self

        class GuardianOpsPlugin(BasePlugin):  # pragma: no cover - needs ADK
            def __init__(self) -> None:
                super().__init__(name="guardianops")

            async def before_tool_callback(self, *, tool, tool_args, tool_context, **_):
                return await guard.before_tool_callback(tool, tool_args, tool_context)

            async def after_tool_callback(self, *, tool, tool_args, tool_context, result, **_):
                return await guard.after_tool_callback(tool, tool_args, tool_context, result)

            async def before_model_callback(self, *, callback_context, llm_request, **_):
                return await guard.before_model_callback(callback_context, llm_request)

            # The agent hooks are not optional decoration: ``after_agent_callback``
            # is where run.end is written and the baseline is persisted. Without
            # them a plugin-mode run never saves what it observed, so novelty
            # fires on every tool forever. ADK hands the plugin the agent as well
            # as the context; the guard only needs the context.
            async def before_agent_callback(self, *, callback_context, **_):
                return await guard.before_agent_callback(callback_context)

            async def after_agent_callback(self, *, callback_context, **_):
                return await guard.after_agent_callback(callback_context)

        return GuardianOpsPlugin()


def sync(callback: Callable) -> Callable:
    """Wrap an async callback for an ADK version that only calls sync hooks.

    Only safe from a thread that is not already running the event loop; if one
    is running, the async callbacks are the correct choice.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(callback(*args, **kwargs))

    return wrapper
