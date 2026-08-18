# GuardianOps — P0

Runtime governance for autonomous AI agents and MCP ecosystems.

This is **P0**: an inline MCP proxy that traces every message, pins tool
definitions, enforces entitlements, and holds high-risk calls for a human. It is
the wedge described in the [concept brief](guardianops.html) — the smallest
thing that is genuinely useful on its own and that the rest of the platform can
grow out of.

Python 3.11+, **standard library only**. No install, no dependencies, no network
fetch.

```
agent / MCP client  ──stdio──▶  guardianops  ──stdio or HTTP──▶  MCP server
                                     │
                                     └──▶  audit ledger · pins · baseline
```

**Full guide: [docs/USAGE.md](docs/USAGE.md)** — install, both integration
paths, every policy key, operations, rollout and troubleshooting.

## Try it

```bash
python3 demo.py                      # the MCP proxy, end to end
python3 demo_adk.py                  # the ADK guard, no ADK needed
python3 -m unittest discover -s tests -v
```

Both demos run from this directory with no install. `demo.py` runs one agent
session three times — shadow, enforce, and again after the upstream server
quietly rewrites a tool's description — then verifies the audit chain and prints
a run report.

## Install

To use it from your own project, install it into that project's environment:

```bash
pip install -e /path/to/guardianops          # editable, so edits take effect
pip install -e "/path/to/guardianops[adk]"   # and pull in google-adk
```

That puts a `guardianops` command on your PATH and makes
`from guardianops.adk import GuardianOpsGuard` importable from anywhere. Without
it, both only work from this directory.

## Use it against a real MCP server

Put the proxy in front of the server in your client's MCP config. Use absolute
paths — the proxy inherits the client's working directory.

```json
{
  "mcpServers": {
    "workspace": {
      "command": "python3",
      "args": [
        "-m", "guardianops", "run",
        "--config", "/abs/path/config.example.json",
        "--",
        "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"
      ]
    }
  }
}
```

For a remote server over Streamable HTTP:

```bash
python3 -m guardianops run --config config.example.json \
  --upstream-url https://host/mcp --header "Authorization: Bearer $TOKEN"
```

## Google ADK

ADK agents get a second, deeper integration. `before_tool_callback` returning a
dict short-circuits the tool and hands that dict to the model as the result —
the exact shape of an inline governance decision — so the policy engine drops
straight in.

```python
from guardianops.adk import GuardianOpsGuard

guard = GuardianOpsGuard(config_path="config.adk.json", app_name="mcp-copilot")

agent = LlmAgent(name="mcp_admin", model="gemini-2.0-flash", tools=[...],
                 **guard.callbacks())
```

For an agent tree you did not build, `guard.install(root_agent)` attaches to it
and its sub-agents. On an ADK with the plugin API,
`Runner(..., plugins=[guard.as_plugin()])` governs every agent at once.

```bash
python3 demo_adk.py             # replays an ADK-shaped run; no ADK needed
python3 examples/adk_agent.py   # the real thing (needs `pip install google-adk`)
```

**What ADK gives you that the MCP proxy cannot:**

- **Every tool.** FunctionTool, AgentTool, OpenAPI toolsets and MCPToolset tools
  all cross the same callback. The proxy only ever sees MCP.
- **Context drift, for real.** `before_model_callback` sees the LlmRequest, so
  the run's objective is observable in-process. This is the fifth signature the
  MCP boundary structurally cannot compute.
- **One run identity.** `invocation_id` ties a planner and its executor into a
  single traced run instead of two disconnected sessions.
- **Governable delegation.** `transfer_to_agent` is an ordinary tool call here,
  so handoff is policy-checked and counts toward autonomy drift.

**What it does not give you: enforcement.** The guard runs inside the agent
process and anyone editing the agent can remove it. It is a fidelity and
coverage layer. Keep the MCP proxy in front of MCP servers, and lock egress at
the network. Use both — `examples/adk_agent.py` wires the proxy in as the
`MCPToolset` command, so MCP tools are governed twice.

Nothing in `guardianops/adk.py` imports ADK at module load, and every ADK object
is read through `getattr` fallbacks, so an ADK version bump that renames a field
degrades to "unknown" instead of crashing your agent.

### Context drift is scored, not enforced, by default

`approval.contextThreshold` defaults to `null`: the score is computed and
recorded, never acted on. That is deliberate. The bundled `LexicalContextScorer`
is token overlap, not semantics — it catches an agent that wandered onto an
unrelated subject and misses one that stayed on topic while doing something
dangerous. Swap in an encoder by passing any `ContextScorer` to the guard.

Building the demo surfaced the failure mode worth knowing about: `transfer_to_agent`
scored **1.00 drift** and got blocked, because delegation arguments are agent
names and never share vocabulary with the objective. Blocking handoff breaks
every multi-agent run. Hence `approval.contextExemptTiers` (default
`["control"]`) — put control-plane tools in a `control` tier and they are scored
on everything except context. Expect to find more of these in your own traffic;
that is what shadow mode is for.

## Commands

| Command | What it does |
|---|---|
| `run` | Proxy an MCP server with governance applied |
| `validate` | Check a policy for errors before it ships; exits non-zero in CI |
| `verify` | Walk the audit ledger's hash chain and report the first break |
| `report` | Summarize runs: outcomes, held calls, peak risk, withheld tools |
| `pins` | Show the pinned tool definitions |

`run` validates the policy at startup and refuses to enforce a broken one.
`verify` takes `--key` (or `--config`) for a signed ledger.

## What it enforces

**Entitlement (privilege drift).** Tools outside `tools.allow` are blocked, and
in enforce mode they are also stripped from `tools/list` so the model never sees
them. Withholding beats refusing: a tool that is not in context is not a tool
the model will keep trying.

**Argument constraints (the resource half of the decision).** Entitlement by tool
name cannot tell `send_message("customer@example.com")` from
`send_message("attacker@evil.example")` —
same tool, same permission, wildly different blast radius. Constraints bound the
values a tool may be pointed at:

```jsonc
"constraints": {
  "read_document": { "doc_id": { "prefix": ["kb/"],
                                 "denyPrefix": ["kb/private/"] } },
  "send_message":  { "to":  { "pattern": "[^@\\s]+@example\\.com", "required": true } },
  "save_memory":   { "key": { "required": true } },
  "run_shell":     { "cmd": { "denyPattern": "rm\\s+-rf" } }
}
```

Matchers are `prefix`, `denyPrefix`, `in`, `pattern` (full match), `denyPattern`
(substring), and `required`. Paths are normalized before prefix matching, so
`/workspace/../etc/passwd` does not satisfy a `/workspace/` rule. Symlinks are
*not* resolved — the target may not exist on the proxy's host — so a prefix rule
bounds the path that was asked for, not necessarily the inode it reaches.

Constraints are checked *before* the operator whitelist: approving
"send_message for this run" approves the tool, not every recipient it could be
pointed at. URLs are exempt from path normalisation, so a `denyPrefix` of
`https://169.254.` keeps matching.

**Blast-radius tiers.** Every tool is `read`, `write`, `destructive`, or
`unclassified`. Unclassified is treated as approval-worthy, not safe — unknown
blast radius is a risk, not the absence of one. Which tiers require approval is
configured in `approval.requireFor`.

**Tool-definition pinning (MCP trust drift).** Each tool's name, description and
input schema are hashed on first sight and pinned. A server that later rewrites
a description — same name, same schema, new instructions to the model — is
caught. Changed definitions are *not* silently re-pinned; an operator approves
the new one explicitly.

**Invocation baseline (tool drift).** First-ever invocation of a tool scores as
novel. Novelty is about invocation, not advertisement: a tool the server has
always offered but the agent has never called is still a first.

**Autonomy drift.** Calls since the last human checkpoint. Past
`approval.autonomyThreshold`, the next call is held regardless of tier.

**Human-in-the-loop.** The intercept panel renders on `/dev/tty`, not stdout —
stdout is the protocol. With no controlling terminal there is no operator, so
`approval.onTimeout` applies. Silence is never treated as consent.

**Content scanning (proxy).** Pinning detects that a description *changed*, so
a server hostile on first contact is pinned in silence, and nothing looks at
tool *results* at all — the one place an injected instruction actually arrives
from outside. `scan.definitions` reads every advertised description on first
sight, including argument descriptions and invisible characters;
`scan.responses` reads what comes back and can `sanitize` the matching spans out
while leaving the rest of the result usable. Defaults are `warn` and `sanitize`,
deliberately not `block`: this is pattern matching against natural language, a
tripwire rather than a barrier.

**Scope (ADK guard).** Asks whether the *request* belongs to this agent at all —
a question context drift cannot answer, since an off-topic request is perfectly
consistent with itself. It runs in `before_model_callback`, so it is the only
control that decides **before inference**: an out-of-scope request costs zero
tokens and reaches nothing. Start at `threshold: null` to score without
enforcing.

**Hash-chained audit ledger.** Every record commits to its predecessor. Editing
or deleting any line invalidates everything after it and `verify` names the
break — and distinguishes a truncated tail (a crash) from an altered record in
the middle (a security event). Set `auditKey` and each record is signed with
HMAC instead of a bare digest, so rewriting the file and recomputing the chain no
longer produces a ledger that verifies. Arguments are redacted by key name and by
value shape before they are written, because the ledger is WORM in production and
a leaked secret cannot be taken back out.

## Risk scoring

The composite uses the weights published in the brief:

```
risk = 0.25·privilege + 0.20·tool + 0.20·mcp_trust + 0.15·autonomy + 0.20·context
```

Which signals are available depends on where you integrate:

| Signal | MCP proxy | ADK guard |
|---|---|---|
| privilege | yes | yes |
| tool (novelty) | yes | yes |
| autonomy | yes | yes, plus delegation depth |
| mcp_trust | yes (definition pinning) | no — pinning lives at the MCP boundary |
| context | **always 0.0** — objective not observable | yes, from the LlmRequest |

Run both and each covers the other's blind spot.

Risk measures *drift*, not blast radius. A destructive tool called exactly as it
always has been scores near zero and is still held for approval, because its
tier says so. The two axes are deliberately separate; conflating them makes a
routine `delete` look like an anomaly and buries the real ones.

## Shadow first

`mode: "shadow"` scores and records everything and enforces nothing, including
tool-list filtering. An inline control that blocks legitimate work gets
uninstalled in a week, so the intended path is: run shadow, read the reports,
tune thresholds against real traffic, then flip to `enforce`. Every shadow record
carries `shadowed: true` and the decision that *would* have been taken.

## Performance

Two costs, measured separately, because they behave very differently.

**The decision** is set membership, integer counters, a weighted sum and (where
configured) constraint matching. `report` prints it per run: **p50 ≈ 6µs**.

**The ledger write** dominates, and how much depends on the durability you ask
for. Measured on APFS:

| `auditSync` | routine call | governance decision |
|---|---|---|
| `always` | 3.0ms | 3.0ms |
| `critical` (default) | **13µs** | **3.0ms** |
| `interval` | 13µs | 13µs |

Records are appended in order and readable immediately in every mode — `fsync`
only decides what survives a power cut, so chain integrity never depends on this
setting. The default syncs records carrying a governance decision (a block, a
hold, a run boundary) and batches the rest.

Note the 3ms: on macOS plain `fsync` returns without flushing the drive's write
cache, so it looks ~70x cheaper than it is while providing no durability
guarantee. Critical records use `F_FULLFSYNC`, which is what actually costs
3ms — paying for the guarantee it appears to be buying. Set `"auditSync":
"interval"` if you would rather batch everything and accept losing the last
couple of seconds on power loss.

These are component numbers, not end-to-end added latency; total proxy overhead
also includes JSON round-tripping and a process hop. Measure it in your own
topology before quoting a number.

## What P0 does not do

Being explicit, because these gaps are load-bearing:

- **It is bypassable.** If an agent can reach the MCP server or the network
  directly, the proxy is advisory. Non-bypassable enforcement is an
  infrastructure control — egress-locked containers with a NetworkPolicy that
  permits egress only via GuardianOps. That is P1.
- **No LLM gateway.** Prompt-level attacks pass straight through. GuardianOps
  catches the *effect* of an injection at the tool boundary, not the injection.
  (The ADK guard sees the objective in-process, which covers context drift, but
  it is not a gateway and does not govern the model call itself.)
- **The ADK guard is in-process and bypassable.** It is coverage, not a boundary.
- **Local state.** The ledger, pins and baseline are files. Production wants
  ClickHouse for traces, Postgres for control plane, S3 Object Lock for the
  ledger.
- **No multi-tenancy, no console, no policy DSL.** One policy governs one proxy.
  Policies can be built programmatically (`Config.from_dict`) as well as loaded
  from JSON, but there are no per-agent or per-tenant policy bundles, and no
  language for expressing rules beyond the config schema.
- **The ledger is only as tamper-proof as the key.** Unsigned, it is
  tamper-*evident*: anyone who can rewrite the file can recompute the whole chain.
  Set `auditKey` and records are signed with HMAC instead, which detects exactly
  that forgery — but the guarantee is the key's, so a key file on the agent host
  beside the ledger it signs buys little. Production wants the key held off-host,
  or records shipped to an append-only sink as they are written.
- **Redaction is a tripwire, not a guarantee.** Credentials are masked by key
  name and by value shape — issuer-prefixed tokens, JWTs, PEM blocks,
  `TOKEN=...` assignments, passwords in connection strings. Pattern matching
  against arbitrary strings catches the recognisable and misses the novel, so a
  secret in a shape nobody has seen still reaches the ledger.
- **Novelty is binary.** No sequence model, so an unremarkable set of tools in a
  bizarre order scores zero on tool drift.
- **Cold start.** A new agent has no baseline, so novelty fires on everything for
  the first run and the deterministic controls carry the load.
- **Server-initiated SSE streams are not handled.** The HTTP transport reads
  responses to POSTed messages, including SSE bodies; a standalone `GET` stream
  opened by the server is not consumed, so server-pushed notifications on that
  channel are missed.

## Layout

```
guardianops/
  policy.py     config, decision engine, drift signals, redaction  ← shared
  audit.py      hash-chained JSONL ledger                          ← shared
  baseline.py   per-agent invocation counts                        ← shared
  hitl.py       /dev/tty intercept panel                           ← shared
  proxy.py      MCP interception engine
  jsonrpc.py    JSON-RPC 2.0 framing for the MCP wire format
  pins.py       trust-on-first-use tool definition pinning
  upstream/     stdio and Streamable HTTP transports
  adk.py        Google ADK callbacks, plugin, scope gate, context scoring
  scan.py       definition and response content scanning
  mock_server.py  a small MCP server for the demo and tests
examples/
  adk_agent.py  a governed ADK agent, both layers wired
tests/          106 tests over policy, pinning, scanning, ledger
                integrity, MCP enforcement, and the ADK guard
```

The engine (`policy`, `audit`, `baseline`, `hitl`) is shared. `proxy.py` and
`adk.py` are two integration points onto the same decisions and the same ledger —
a run governed at both layers reads as one chain.
