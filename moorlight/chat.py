"""Built-in chat: runs headless Claude Code (`claude -p`) with the moorlight MCP tools attached.

Every message is sent with a compact description of the current view (visible maps,
contour levels, centre, selection), so "this map" or "these atoms" resolve to what is
on screen. The Claude Code session is resumed between messages, so it remembers the
conversation; it can also be resumed from a terminal with `claude --resume <id>`.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time

SYSTEM = """You are embedded in "Moorlight", a lightweight Coot-like model-building viewer used for
crystallographic (mostly time-resolved, TorchRef difference-refinement) work. The user sees a 3D view of
models and maps. Use the mcp__moorlight__* tools to inspect and change what they see:
- state: what is loaded/visible (maps with contour levels in sigma, centre, selected atoms, refinement status)
- screenshot: returns an image of the current view
- goto / set_map / peaks / map_correlation / density_at_atoms for analysis
- set_refinement_options: target "map" (2mFo-DFc-like map), "difference" (fit light-minus-dark model density
  to a difference map such as WDF, dark reference = dark model / other altlocs / start coordinates) or "both";
  non-bonded repulsion is off by default
- refine, add_altconf, rotamers/set_rotamer, mutate, add_water, add_ion, add_ligand, add_residue, renumber,
  delete, fix_atoms, undo for building. Model edits are undoable.
Each user message starts with a [viewer] block describing the current view; "this map", "here" or "these
atoms" refer to it. Map numbers like [3] are map ids. Prefer showing the user things (goto, set_map) over
long explanations. When you run TorchRef refinements, write outputs to a new directory and afterwards load
them with load_torchref so the user can inspect them. Keep answers short; the chat panel is narrow."""


def _short_state(s):
    lines = []
    for m in s.molecules.values():
        lines.append(f"molecule {m.id}: {m.name} ({m.path}){' [active]' if m.id == s.active_mol else ''}"
                     f"{'' if m.visible else ' [hidden]'}")
    for x in s.maps.values():
        if x.visible:
            lines.append(f"map [{x.id}] {x.name} from {x.source}: shown at {x.level_sigma:.2f} sigma"
                         f"{' (+/-)' if x.kind == 'diff' else ''}{' [refinement map]' if x.id == s.refinement_map else ''}")
    hidden = [f"[{x.id}] {x.name}" for x in s.maps.values() if not x.visible]
    if hidden:
        lines.append("hidden maps: " + ", ".join(hidden))
    st = s.state()
    v = st["view"]
    c = v.get("center") or [0, 0, 0]
    lines.append(f"view centre: ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})" +
                 (f", nearest atom {v['nearest_atom']}" if v.get("nearest_atom") else ""))
    if v.get("selection_atoms"):
        lines.append("selected atoms: " + ", ".join(v["selection_atoms"][:20]))
    if st.get("refinement"):
        lines.append(f"real-space refinement running on {st['refinement']['moving_atoms']} atoms")
    return "\n".join(lines)


class ClaudeChat:
    def __init__(self, session, emit, claude_bin=None, allow_bash=False, model=None, port=None):
        self.s = session
        self.emit = emit
        self.claude = claude_bin or shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
        self.allow_bash = allow_bash
        self.model = model
        self.port = port
        self.session_id = None
        self.proc = None
        self.history: list[dict] = []

    def settings(self):
        return {"allow_bash": self.allow_bash, "model": self.model, "session_id": self.session_id,
                "claude": self.claude}

    def update_settings(self, d):
        if "allow_bash" in d:
            self.allow_bash = bool(d["allow_bash"])
        if "model" in d:
            self.model = d["model"] or None

    def reset(self):
        self.session_id = None
        self.history.clear()

    def _mcp_config(self):
        d = os.path.join(self.s.workdir, ".moorlight")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "mcp.json")
        cfg = {"mcpServers": {"moorlight": {
            "command": sys.executable, "args": ["-m", "moorlight.mcp", "--url", f"http://127.0.0.1:{self.port}"],
            "env": {"PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}}}}
        with open(p, "w") as fh:
            json.dump(cfg, fh, indent=1)
        return p

    async def _out(self, msg):
        self.history.append(msg)
        await self.emit(msg)

    async def stop(self):
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()

    async def send(self, text):
        if self.proc and self.proc.returncode is None:
            await self._out({"type": "chat", "role": "system", "text": "Claude is still busy – wait or press stop."})
            return
        await self._out({"type": "chat", "role": "user", "text": text})
        prompt = f"[viewer]\n{_short_state(self.s)}\n[/viewer]\n\n{text}"
        tools = ["mcp__moorlight", "Read", "Glob", "Grep", "TodoWrite"]
        if self.allow_bash:
            tools += ["Bash", "Write", "Edit"]
        args = [self.claude, "-p", prompt, "--output-format", "stream-json", "--verbose",
                "--include-partial-messages", "--mcp-config", self._mcp_config(),
                "--append-system-prompt", SYSTEM, "--allowedTools", *tools]
        if self.session_id:
            args += ["--resume", self.session_id]
        if self.model:
            args += ["--model", self.model]
        t0 = time.time()
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *args, cwd=self.s.workdir, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=16 * 1024 * 1024)
        except FileNotFoundError:
            await self._out({"type": "chat", "role": "system", "text": f"claude CLI not found at {self.claude}"})
            return
        await self.emit({"type": "chat_busy", "busy": True})
        streamed = False
        try:
            async for raw in self.proc.stdout:
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                et = ev.get("type")
                if et == "system" and ev.get("subtype") == "init":
                    self.session_id = ev.get("session_id", self.session_id)
                elif et == "stream_event":
                    e = ev.get("event", {})
                    if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta":
                        streamed = True
                        await self.emit({"type": "chat_delta", "text": e["delta"]["text"]})
                    elif e.get("type") == "message_start":
                        await self.emit({"type": "chat_delta_start"})
                elif et == "assistant":
                    for block in ev.get("message", {}).get("content", []):
                        if block.get("type") == "text" and block.get("text", "").strip():
                            await self._out({"type": "chat", "role": "assistant", "text": block["text"],
                                             "streamed": streamed})
                        elif block.get("type") == "tool_use":
                            name = block.get("name", "").replace("mcp__moorlight__", "")
                            if name == "ToolSearch":
                                continue
                            await self._out({"type": "chat", "role": "tool", "text": name,
                                             "input": json.dumps(block.get("input", {}))[:300]})
                    streamed = False
                elif et == "user":
                    for block in ev.get("message", {}).get("content", []) or []:
                        if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                            c = block.get("content")
                            c = c if isinstance(c, str) else json.dumps(c)[:300]
                            await self._out({"type": "chat", "role": "tool_error", "text": c[:300]})
                elif et == "result":
                    self.session_id = ev.get("session_id", self.session_id)
                    if ev.get("is_error"):
                        await self._out({"type": "chat", "role": "system", "text": str(ev.get("result"))[:500]})
                    cost = ev.get("total_cost_usd")
                    await self.emit({"type": "chat_done", "seconds": round(time.time() - t0, 1),
                                     "cost": cost, "session_id": self.session_id})
            await self.proc.wait()
            if self.proc.returncode not in (0, None, -15):
                err = (await self.proc.stderr.read()).decode()[-800:]
                await self._out({"type": "chat", "role": "system", "text": f"claude exited {self.proc.returncode}: {err}"})
        finally:
            await self.emit({"type": "chat_busy", "busy": False})
