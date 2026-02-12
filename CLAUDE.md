# xBridge MCP

## Project
Python MCP server exposing xAI Grok API (chat, web search, X search, image gen, video gen, sessions, chains). Branded as **xBridge MCP**, independent from xAI.

## Stack
- Python 3.10+ | mcp>=1.0.0 | httpx>=0.27.0
- Build: hatchling | Entry: `xbridge-mcp = xbridge_mcp.server:run`
- Env: `XAI_API_KEY` required
- Docker: `hrco/xbridge-mcp:latest`

## Structure
```
xbridge_mcp/
  server.py           # MCP server + all 16 tool handlers
  session_manager.py  # JSON-file session persistence
  tool_chains.py      # Chain execution framework
```

## Patterns
- All xAI API calls go through `make_grok_request()` in server.py
- Tools registered via `@server.list_tools()` / `@server.call_tool()`
- Sessions stored as JSON files in `.grok_sessions/`
- Response parsing: `extract_response_text()` handles nested output
- MCP tool names keep `grok-*` prefix (xAI API tools, not our brand)

## Monetization
- Free tier: MIT source on GitHub, `pip install` from source
- Pro tier: $3.69/mo via Gumroad, pre-built Docker image `hrco/xbridge-mcp`
- BYOK model: users always bring their own XAI_API_KEY

## Dev Context
- `GROK-DEV-CONTEXT.md` - Session memory and architecture notes
