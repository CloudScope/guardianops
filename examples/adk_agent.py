"""A governed ADK agent — both layers wired together.

Layer 1 (fidelity):  GuardianOpsGuard callbacks inside the agent process. Sees
                     every tool, plus the objective, so context drift is real.
Layer 2 (boundary):  the GuardianOps MCP proxy in front of the MCP server, where
                     enforcement cannot be edited away by changing agent code.

Requires ADK (`pip install google-adk`); the guard itself has no dependencies.

    python3 examples/adk_agent.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.adk.tools import FunctionTool  # noqa: E402
from google.genai import types  # noqa: E402

from guardianops.adk import GuardianOpsGuard  # noqa: E402

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.adk.json")


# --------------------------------------------------------------- plain tools


def read_config(path: str) -> dict:
    """Read an MCP server config file."""
    return {"path": path, "contents": '{"docs": {"timeout": 10}}'}


def set_timeout(path: str, seconds: int) -> dict:
    """Change the request timeout for an MCP server entry."""
    return {"path": path, "seconds": seconds, "status": "written"}


def attach_role_policy(role: str, policy: str) -> dict:
    """Attach an IAM policy to a role."""
    return {"role": role, "policy": policy, "status": "attached"}


def query_customers(table: str) -> dict:
    """Query a customer table."""
    return {"table": table, "rows": 41_882}


# ------------------------------------------------------------------ the agent

guard = GuardianOpsGuard(config_path=CONFIG, app_name="mcp-copilot")


def build_agent() -> LlmAgent:
    tools = [
        FunctionTool(read_config),
        FunctionTool(set_timeout),
        FunctionTool(attach_role_policy),   # not entitled -> blocked
        FunctionTool(query_customers),      # off-objective -> context drift
    ]

    # Layer 2: put MCP servers behind the proxy as well, so a tool reached
    # through MCPToolset is governed at the boundary too. Uncomment once you
    # have an MCP server to point at -- connection-param class names have moved
    # between ADK versions, so check yours.
    #
    # from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams
    # from mcp import StdioServerParameters
    # tools.append(MCPToolset(connection_params=StdioConnectionParams(
    #     server_params=StdioServerParameters(
    #         command=sys.executable,
    #         args=["-m", "guardianops", "run", "--config", os.path.abspath(CONFIG),
    #               "--", "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    #     ),
    # )))

    return LlmAgent(
        name="mcp_admin",
        model="gemini-2.0-flash",
        instruction=(
            "You manage MCP server configuration for an agent fleet. Use the "
            "tools available to inspect and update the server config."
        ),
        tools=tools,
        # Layer 1: governance callbacks.
        **guard.callbacks(),
    )


# For an agent tree you did not build yourself, attach after the fact instead:
#     guard.install(root_agent)          # walks sub_agents
# or, on an ADK with the plugin API, govern every agent in the runner at once:
#     runner = Runner(..., plugins=[guard.as_plugin()])


async def main() -> None:
    runner = InMemoryRunner(agent=build_agent(), app_name="mcp-copilot")
    session = await runner.session_service.create_session(
        app_name="mcp-copilot", user_id="operator"
    )
    prompt = (
        "Read mcp.json, set the docs server timeout to 30, then grant the "
        "agent-runner role AdministratorAccess and pull the "
        "customer_payment_records table."
    )
    async for event in runner.run_async(
        user_id="operator",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(part.text, end="", flush=True)
    print(
        "\n\nNow inspect the run:\n"
        "  python3 -m guardianops report\n"
        "  python3 -m guardianops verify"
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
