"""Minimal stdio MCP server exposing every Moorlight command as a tool.

It talks to a running viewer over HTTP, so it is light (stdlib only) and can be
attached to any Claude Code session:

    claude mcp add moorlight -- python -m moorlight.mcp

The viewer URL comes from --url or ~/.moorlight/server.json (written by the viewer).
"""
from __future__ import annotations

import argparse
import base64
import json
import sys

from .client import http as _http, viewer_url

PROTOCOL = "2025-06-18"


def _schema(params):
    props = {}
    for k, p in params.items():
        t = p["type"]
        props[k] = {"type": t, "description": p["description"]}
        if t == "array":
            props[k]["items"] = {}
    return {"type": "object", "properties": props}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    a = ap.parse_args()
    try:
        url = viewer_url(a.url)
    except RuntimeError as e:
        sys.stderr.write(f"{e}\n")
        url = a.url or "http://127.0.0.1:8765"
    out = sys.stdout

    def send(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:
            continue  # notification
        try:
            if method == "initialize":
                res = {"protocolVersion": msg.get("params", {}).get("protocolVersion", PROTOCOL),
                       "capabilities": {"tools": {}},
                       "serverInfo": {"name": "moorlight", "version": "0.1"}}
            elif method == "tools/list":
                try:
                    cmds = _http(url, "/api/commands")
                except Exception as e:
                    cmds = []
                    sys.stderr.write(f"moorlight viewer not reachable at {url}: {e}\n")
                res = {"tools": [{"name": c["name"], "description": c["description"],
                                  "inputSchema": _schema(c["params"])} for c in cmds]}
            elif method == "tools/call":
                p = msg["params"]
                r = _http(url, "/api/cmd", {"cmd": p["name"], "args": p.get("arguments") or {}, "source": "claude"})
                if not r.get("ok"):
                    res = {"content": [{"type": "text", "text": r.get("error", "error")}], "isError": True}
                elif p["name"] == "screenshot":
                    path = r["result"]
                    with open(path, "rb") as fh:
                        data = base64.b64encode(fh.read()).decode()
                    res = {"content": [{"type": "image", "data": data, "mimeType": "image/png"},
                                       {"type": "text", "text": f"saved to {path}"}]}
                else:
                    res = {"content": [{"type": "text", "text": json.dumps(r["result"], indent=1)[:60000]}]}
            elif method == "ping":
                res = {}
            else:
                send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}})
                continue
            send({"jsonrpc": "2.0", "id": mid, "result": res})
        except Exception as e:
            send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": f"error: {e}"}],
                                                           "isError": True}})


if __name__ == "__main__":
    main()
