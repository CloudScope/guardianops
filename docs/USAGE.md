# Using GuardianOps

End to end: install it, put it in front of something, write a policy, read what
it recorded. Every command here has been run against this repo.

- [What it is](#what-it-is)
- [Install](#install)
- [Sixty seconds](#sixty-seconds)
- [Path A — the MCP proxy](#path-a--the-mcp-proxy)
- [Path B — the ADK guard](#path-b--the-adk-guard)
- [Both at once](#both-at-once)
- [Writing a policy](#writing-a-policy)
- [Where each control fires](#where-each-control-fires)
- [Operating it](#operating-it)
- [Rollout](#rollout)
- [Using it as a library](#using-it-as-a-library)
- [What it does not do](#what-it-does-not-do)
- [Troubleshooting](#troubleshooting)

---

## What it is

Two ways onto the same decision engine, the same policy format and the same
ledger.

```
                    ┌─ Path B: in-process, sees the objective and every tool
                    │
agent / MCP client ─┴─ stdio ─▶ guardianops ─ stdio|HTTP ─▶ MCP server
                                     │        └─ Path A: the boundary
                                     └──▶ ledger · pins · baseline
```

**Path A, the proxy**, is the boundary. It is a separate process, so it survives
someone editing the agent, and it is the only layer that sees tool definitions
and tool results.

**Path B, the ADK guard**, is coverage. It runs inside the agent, so it sees the
user's request and every tool including non-MCP ones — but anyone who can edit
the agent can remove it.

Neither is sufficient alone. Each covers the other's blind spot.

---

## Install

```bash
pip install guardianops              # stdlib only, no dependencies
pip install "guardianops[adk]"       # adds google-adk and mcp for Path B
```

Python 3.11+. Installing puts a `guardianops` command on your PATH.

From a checkout, `pip install -e .` — or skip installing entirely and run
`python3 -m guardianops` from the repo root.

---

## Sixty seconds

```bash
python3 demo.py        # the proxy: shadow, enforce, then a rug-pull
python3 demo_adk.py    # the ADK guard, replayed without ADK installed
```

`demo.py` drives a real agent session through a real proxy subprocess against a
mock MCP server, three times, then verifies the ledger. It is the fastest way to
see every control fire.

---

## Path A — the MCP proxy

You insert the proxy by editing your MCP client's config. The agent never knows.

```json
{
  "mcpServers": {
    "workspace": {
      "command": "python3",
      "args": [
        "-m", "guardianops", "run",
        "--config", "/abs/path/to/policy.json",
        "--",
        "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"
      ]
    }
  }
}
```

Everything after `--` is the command that would have launched the server
directly. Use an absolute path for `--config`: the proxy inherits the client's
working directory, which is rarely the one you had in mind.

A remote server over Streamable HTTP instead:

```bash
guardianops run --config policy.json \
  --upstream-url https://host/mcp --header "Authorization: Bearer $TOKEN"
```

Useful flags: `--mode shadow|enforce` overrides the policy without editing it,
`--run-id` labels every record from this run, `--audit` and `--baseline`
override paths.

### Driving it by hand

The proxy speaks JSON-RPC on stdin/stdout, so you can test without a client —
but you must hold the pipe open and read each response. Piping with `printf`
closes stdin before the server replies and nothing gets pinned. See
`examples/` for a driver, or just run `demo.py`.

---

## Path B — the ADK guard

```python
from guardianops.adk import GuardianOpsGuard

guard = GuardianOpsGuard.from_policy("policies/agents/chat.json")
agent = LlmAgent(name="assistant", model="gemini-flash-latest",
                 tools=[...], **guard.callbacks())
```

Name, mode, ledger, pins and baseline all come from the policy file, so adding
an agent is adding a JSON file.

Three ways to attach, in ascending coverage:

```python
agent = LlmAgent(..., **guard.callbacks())   # one agent
guard.install(root_agent)                    # an agent tree you did not build
Runner(..., plugins=[guard.as_plugin()])     # every agent in the runner
```

`as_plugin()` is the one to prefer: a sub-agent added later is governed without
anyone remembering to wire it up.

### With `adk web`

The ADK loader looks for a module-level `app` **before** `root_agent`, and `App`
takes a plugin list — so the dev UI can be governed rather than being a hole in
the policy:

```python
# agents/assistant/agent.py
from google.adk.apps import App

root_agent = build_agent()
app = App(name="assistant", root_agent=root_agent,
          plugins=[build_guard().as_plugin()])
```

```bash
adk web agents
```

Export only `root_agent` and the UI runs your agent **ungoverned**.

---

## Both at once

Point ADK's `McpToolset` at the proxy instead of at the server. The toolset
believes the proxy is the server:

```python
McpToolset(connection_params=StdioConnectionParams(
    server_params=StdioServerParameters(
        command=sys.executable,
        args=["-m", "guardianops", "run", "--config", "policies/servers/time.json",
              "--", sys.executable, "-m", "mcp_server_time"])))
```

Now definition pinning, scanning and entitlement apply at a boundary outside the
agent process, while the in-process guard adds the objective and context drift.

---

## Writing a policy

One file per agent, and one per MCP server. A shared file entitles every agent
to the union of every agent's tools, which is the opposite of least privilege.

Full example with every section:

```jsonc
{
  "name": "support-bot",              // identity; defaults to the filename
  "mode": "shadow",                   // "shadow" | "enforce"
  "defaultDecision": "allow",         // what a call matching no rule gets

  // Relative paths resolve next to THIS FILE, not the working directory.
  "auditPath":    ".guardianops/ledger/agent-support.jsonl",
  "baselinePath": ".guardianops/baseline.json",
  "filterToolList": true,             // strip non-entitled tools from tools/list

  "tools": {
    "allow": ["search_docs", "read_document", "send_message"],
    "deny":  [],
    "tiers": {
      "read":        ["search_docs", "read_document"],
      "write":       ["save_memory"],
      "destructive": ["send_message"]
    },
    "constraints": {
      "read_document": { "doc_id": { "prefix": ["kb/"],
                                     "denyPrefix": ["kb/private/"] } },
      "send_message":  { "to": { "pattern": "[^@\\s]+@example\\.com",
                                 "required": true } },
      "run_shell":     { "cmd": { "denyPattern": "rm\\s+-rf" } }
    }
  },

  "approval": {
    "requireFor": ["destructive", "unclassified"],
    "timeoutSeconds": 120,
    "onTimeout": "block",             // silence is never consent
    "autonomyThreshold": 25,          // calls since the last human checkpoint
    "contextThreshold": null,         // null = score context drift, never act
    "contextExemptTiers": ["control"]
  },

  "pin": {
    "path": ".guardianops/pins.json",
    "onChange": "escalate"            // "block" | "escalate" | "warn"
  },

  "scan": {                           // proxy only
    "definitions": "warn",            // "block" | "warn" | "log" | ""
    "responses":   "sanitize"         // "block" | "sanitize" | "log" | ""
  },

  "scope": {                          // ADK guard only
    "terms": ["ticket", "refund", "order", "account"],
    "threshold": 0.9,                 // null = score without enforcing
    "denyPatterns": ["\\bmedical advice\\b"],
    "onOutOfScope": "block",
    "message": "That is outside what this agent handles."
  },

  "redactKeys": ["password", "secret", "token", "api_key", "credential"]
}
```

### The closed allowlist

`tools.allow` **is** the default-deny mechanism. Anything not named is blocked
as privilege drift, including a tool the agent genuinely has. To grant a tool,
add its name; to revoke it, remove the name.

`deny` is for carving an exception out of something otherwise permitted — it is
not how you express "block everything".

> **Do not set `"defaultDecision": "block"`** to mean default-deny. It blocks
> explicitly-allowed, in-tier, constraint-clean tools too, and it is
> self-locking: blocked calls never update the baseline, so every tool stays
> permanently novel and permanently blocked.

### Argument constraints

Entitlement by name cannot tell `send_message("customer@example.com")` from
`send_message("attacker@evil.example")`. Matchers: `prefix`, `denyPrefix`, `in`,
`pattern` (full match), `denyPattern` (substring), `required`.

Filesystem paths are normalised before prefix matching, so
`/workspace/../etc/passwd` does not satisfy a `/workspace/` rule. URLs are
exempt from normalisation — collapsing `https://` to `https:/` would make a
`denyPrefix` of `https://169.254.` silently stop matching.

### Scope — the pre-inference gate

Scope asks whether the **request** belongs to this agent at all. Context drift
cannot answer that: it scores a tool call against the request, and an off-topic
request is perfectly consistent with itself.

It is the only control that fires **before** the model is called, so an
out-of-scope request costs zero tokens.

Start with `"threshold": null`, which scores into the ledger without refusing
anyone, and tune against real phrasings. In-scope questions typically land
0.3–0.8; unrelated ones land 1.0. Remember to list greeting words — otherwise
"hello" scores 1.0 and gets turned away.

### Scanning — definitions and responses

MCP hands the model two kinds of text that read like instructions:

- **definitions** — a tool's description enters the context window verbatim.
  Pinning only detects *change*, so a server hostile on first contact is pinned
  in silence. The scanner reads the text instead of only hashing it, including
  argument descriptions and invisible characters.
- **responses** — a tool result is data from outside that lands in the context
  window. No other control looks at it.

Defaults are `warn` and `sanitize`, deliberately not `block`. This is pattern
matching against natural language: a tripwire, not a barrier.

---

## Where each control fires

| Control | Layer | Fires | Token cost |
|---|---|---|---|
| Scope (terms, denyPatterns) | ADK guard | before inference | **zero** |
| Definition scan | proxy | on `tools/list` | none |
| Entitlement | both | on the call | +1 round trip |
| Argument constraints | both | on the call | +1 round trip |
| Tier approval | both | on the call | +1 round trip |
| Definition pinning | proxy | on `tools/list` | none |
| Response scan | proxy | after the call | none |
| Context drift | ADK guard | on the call | +1 round trip |
| Autonomy drift | both | on the call | +1 round trip |

A blocked tool call does **not** save tokens — it costs more. A tool-calling
turn is two model calls (one to choose the tool, one to speak after seeing its
result), and a block *is* that second call, with the block message added to the
prompt. Only the pre-inference gate is free.

---

## Operating it

```bash
guardianops report --audit .guardianops/ledger/     # a directory, or one file
guardianops verify --audit .guardianops/ledger/
guardianops pins --show
```

### One ledger per writer

A hash chain has exactly one writer by construction. Two processes appending to
one file interleave into a chain that cannot verify — and a multi-server agent
runs several proxies at once. Give each writer its own file:

```
.guardianops/ledger/
  agent-support.jsonl
  server-mcp-time.jsonl
  server-mcp-fetch.jsonl
```

`verify` checks each chain independently; `report` correlates across all of
them. Pins and baseline are keyed by server and agent identity, so those are
safe — and better — shared.

### Reading the ledger

Every record commits to its predecessor. Editing or deleting a line invalidates
everything after it, and `verify` names the break — distinguishing a truncated
tail (a crash) from an altered record in the middle (a security event).
Credential-shaped argument keys are redacted before they are written.

---

## Rollout

1. **Shadow, with a closed allowlist from day one.** Entitlement is the one
   control you can get right before seeing traffic — you know which tools you
   gave the agent.
2. **Read `report` after real usage.** Look at what was withheld and what peaked
   on risk.
3. **Add constraints for tools that touch resources** — paths, URLs, recipients.
   This is where most of the value is.
4. **Then enforce.** Keep `defaultDecision: allow`; the allowlist is your
   default-deny.
5. **Tier the destructive things last**, once you know who is around to approve
   them. If a human is asked ten times a session they will start approving
   reflexively.

---

## Using it as a library

`__init__.py` exports only `__version__`; import from submodules.

```python
from guardianops import policy, scan
from guardianops.policy import Config, evaluate, redact
from guardianops.audit  import AuditLog, verify, read_all
from guardianops.pins   import PinStore, digest
from guardianops.baseline import InvocationBaseline
from guardianops.adk    import GuardianOpsGuard, scope_drift
```

`evaluate` is pure — no I/O, no state — so it drops into any agent loop:

```python
cfg = Config.load("policy.json")
v = evaluate(cfg, "send_message", novel=True, pin_changed=False,
             calls_since_approval=0, arguments={"to": "x@example.com"})
v.decision   # what policy concluded
v.effective  # what to actually do (shadow downgrades to allow)
v.reason, v.tier, v.risk, v.signals.as_dict()
```

To surface decisions somewhere a human is already looking, set an observer:

```python
guard.on_decision = lambda record: dashboard.push(record)
```

It cannot change the outcome, and an observer that raises is logged and
swallowed — governance already happened by the time it runs.

---

## What it does not do

- **It is bypassable.** The guard runs inside the agent; the proxy is advisory
  if the agent can reach the network directly. Non-bypassable enforcement is an
  infrastructure control — egress locked so GuardianOps is the only route out.
- **No LLM gateway.** Prompt-level attacks pass through. Scope and scanning are
  heuristics, not comprehension.
- **Scanning is pattern matching.** It catches the careless and the reused,
  misses the novel and the paraphrased.
- **Local state.** Ledger, pins and baseline are files. Production wants an
  append-only sink and a key held outside the agent host.
- **The ledger is tamper-evident, not tamper-proof.** Anyone who can write the
  file can recompute the chain.
- **No kill switch.** A human can decline one call but cannot terminate a run.
- **Cold start.** A new agent has no baseline, so novelty fires on everything
  for the first run.

---

## Troubleshooting

**A policy edit did nothing.** You are in shadow mode. `deny`, tiers and
constraints are all scored and recorded, and none of them act.

**Everything is blocked, forever.** `defaultDecision` is `block`. See the
warning above — it is self-locking.

**`verify` says LEDGER TAMPERED but nobody touched it.** Two processes are
writing to one ledger. Give each writer its own file.

**`adk web` says "No root_agent found".** Your agent package name collides with
a module it imports. The loader puts the agents directory on `sys.path` and
imports by folder name, then swallows the resulting `ImportError`. Rename the
package.

**One off-topic message refused the rest of the session.** Fixed in this
version — ADK sends the whole conversation, and the objective was being read
from the first user turn. It now reads the latest.

**`ModuleNotFoundError: mcp.shared.session`.** mcp 2.0 moved modules that ADK
still imports. Pin `mcp<2`; the `[adk]` extra already does.

**429 from Gemini with `limit: 0`.** Free tier is not enabled on that project.
A `PerMinute` quota clears in about a minute; `PerDay` does not clear until the
daily reset, and it is per model — switching models buys another allowance.

**The token count is blank on a refused turn.** There was no model call. Blank
is not a missing number; it is zero.
