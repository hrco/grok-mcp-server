# GROK MCP Server

## Project
Python MCP server exposing xAI Grok API (chat, web search, X search, sessions, chains).

## Stack
- Python 3.10+ | mcp>=1.0.0 | httpx>=0.27.0
- Build: hatchling | Entry: `grok-mcp = grok_mcp_server.server:main`
- Env: `XAI_API_KEY` required

## Structure
```
grok_mcp_server/
  server.py           # MCP server + all tool handlers
  session_manager.py  # JSON-file session persistence
  tool_chains.py      # Chain execution framework
```

## Patterns
- All xAI API calls go through `make_grok_request()` in server.py
- Tools registered via `@server.list_tools()` / `@server.call_tool()`
- Sessions stored as JSON files in `.grok_sessions/`
- Response parsing: `extract_response_text()` handles nested output

## Agent Delegation
| Agent | File | Purpose |
|-------|------|---------|
| grok-mcp-expert | `.claude/agents/grok-mcp-expert.md` | xAI API research, upgrade planning, image gen integration |

## Dev Context
- `GROK-DEV-CONTEXT.md` - Session memory and architecture notes
