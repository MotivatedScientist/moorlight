"""aiohttp server: static viewer, command API, map meshes, websocket events, chat."""
from __future__ import annotations

import asyncio
import json
import os
import traceback

import numpy as np
from aiohttp import web, WSMsgType

from .maps import pack_meshes
from .session import COMMANDS, Session
from .chat import ClaudeChat

WEB = os.path.join(os.path.dirname(__file__), "web")
STATE_FILE = os.path.expanduser("~/.moorlight/server.json")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def dumps(o):
    return json.dumps(o, default=_json_default)


class App:
    def __init__(self, session: Session, chat_opts=None):
        self.s = session
        self.clients: set[web.WebSocketResponse] = set()
        self.chat = ClaudeChat(session, self.broadcast_async, **(chat_opts or {}))
        session.broadcast = self.broadcast

    # ------------------------------------------------------------- events
    def broadcast(self, msg):
        loop = self.s.loop
        if loop is None:
            return
        data = dumps(msg)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            loop.create_task(self._send_all(data))
        else:
            asyncio.run_coroutine_threadsafe(self._send_all(data), loop)

    async def broadcast_async(self, msg):
        await self._send_all(dumps(msg))

    async def _send_all(self, data):
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except Exception:
                self.clients.discard(ws)

    # ------------------------------------------------------------- routes
    async def index(self, request):
        return web.FileResponse(os.path.join(WEB, "index.html"))

    async def commands(self, request):
        return web.json_response([
            {"name": n, "description": c["desc"],
             "params": {k: {"type": t, "description": d} for k, (t, d) in c["params"].items()}}
            for n, c in COMMANDS.items()])

    async def cmd(self, request):
        body = await request.json()
        name, args = body.get("cmd"), body.get("args") or {}
        src = body.get("source", "ui")
        try:
            result = await asyncio.to_thread(self.s.run, name, args)
            if src != "ui":
                self.broadcast({"type": "remote_cmd", "cmd": name, "args": args})
            return web.Response(text=dumps({"ok": True, "result": result}), content_type="application/json")
        except Exception as e:
            traceback.print_exc()
            self.s.say(f"error in {name}: {e}")
            return web.Response(text=dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}),
                                content_type="application/json")

    async def full_state(self, request):
        s = self.s
        return web.Response(text=dumps({
            "molecules": [m.to_json() for m in s.molecules.values()],
            "maps": [x.info() for x in s.maps.values()],
            "refinement_map": s.refinement_map, "active_mol": s.active_mol,
            "view": s.view, "log": s.log[-50:], "chat": self.chat.history[-100:],
            "chat_settings": self.chat.settings(),
            "rsr_settings": s.rsr_settings(),
            "rsr": None if s.rsr is None else {"mol": s.rsr.mol.id, "idx": s.rsr.moving_mol.tolist()},
        }), content_type="application/json")

    async def mesh(self, request):
        q = request.query
        x = self.s.maps.get(int(request.match_info["id"]))
        if x is None:
            raise web.HTTPNotFound()
        center = [float(q["x"]), float(q["y"]), float(q["z"])]
        radius = float(q.get("r", x.radius))
        level = float(q["level"]) if "level" in q else None
        meshes = await asyncio.to_thread(x.mesh, center, radius, level)
        body = pack_meshes({"map": x.id, "center": center, "level_sigma": level or x.level_sigma}, meshes)
        return web.Response(body=body, content_type="application/octet-stream")

    async def ws(self, request):
        ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024)
        await ws.prepare(request)
        self.clients.add(ws)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    await self.on_ws(json.loads(msg.data))
                except Exception as e:
                    traceback.print_exc()
                    await ws.send_str(dumps({"type": "log", "text": f"error: {e}"}))
        finally:
            self.clients.discard(ws)
        return ws

    async def on_ws(self, m):
        t = m.get("type")
        s = self.s
        if t == "view":
            for k in ("center", "zoom", "slab", "selection", "clicked", "active_map"):
                if k in m:
                    s.view[k] = m[k]
            for mid, mv in (m.get("maps") or {}).items():
                x = s.maps.get(int(mid))
                if x:
                    x.visible = mv.get("visible", x.visible)
                    x.level_sigma = mv.get("level_sigma", x.level_sigma)
        elif t == "pull":
            s.pull(m["atom"], m.get("xyz"))
        elif t == "reply":
            s.reply(m["id"], m.get("data"))
        elif t == "refine_atoms":
            try:
                await asyncio.to_thread(s.refine_atoms, m["idx"], m.get("altloc"))
            except Exception as e:
                s.say(f"error: {e}")
        elif t == "fix":
            await asyncio.to_thread(s.fix_indices, m["idx"], m.get("fix", True))
        elif t == "chat":
            asyncio.create_task(self.chat.send(m["text"]))
        elif t == "chat_stop":
            await self.chat.stop()
        elif t == "chat_settings":
            self.chat.update_settings(m.get("settings") or {})
            await self.broadcast_async({"type": "chat_settings", "settings": self.chat.settings()})
        elif t == "chat_new":
            self.chat.reset()
            await self.broadcast_async({"type": "chat_reset"})

    def make_app(self):
        @web.middleware
        async def no_stale_ui(request, handler):
            # Revalidate the viewer's HTML/JS/CSS each load so an updated Moorlight is never shadowed by cache.
            resp = await handler(request)
            if request.path == "/" or request.path.startswith("/static/"):
                resp.headers["Cache-Control"] = "no-cache"
            return resp

        app = web.Application(client_max_size=64 * 1024 * 1024, middlewares=[no_stale_ui])
        app.router.add_get("/", self.index)
        app.router.add_get("/api/commands", self.commands)
        app.router.add_post("/api/cmd", self.cmd)
        app.router.add_get("/api/state", self.full_state)
        app.router.add_get("/api/map/{id}/mesh", self.mesh)
        app.router.add_get("/ws", self.ws)
        app.router.add_static("/static", WEB)

        async def on_startup(app):
            self.s.loop = asyncio.get_running_loop()

        app.on_startup.append(on_startup)
        return app


def write_state_file(url, workdir):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as fh:
        json.dump({"url": url, "pid": os.getpid(), "workdir": workdir}, fh)
