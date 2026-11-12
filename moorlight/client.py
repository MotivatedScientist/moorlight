"""Talk to a running viewer over HTTP (stdlib only): used by `moorlight cmd` and the MCP server.

    moorlight cmd --list                         # commands and their parameters
    moorlight cmd state compact=true
    moorlight cmd goto chain=A resseq=85 mol=2
    moorlight cmd frame_atoms atoms='["A/85 SG", "B/501 C1"]'
    moorlight cmd density_profile a="A/85 SG" b="B/501 C1" maps=3,4
    moorlight cmd load_mtz '{"path": "x.mtz", "label": "9DG2"}'

Values are read as JSON when possible (numbers, true/false, lists), otherwise as strings. The result is printed as JSON.
The viewer URL comes from --url, $MOORLIGHT_URL or ~/.moorlight/server.json.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

STATE_FILE = os.path.expanduser("~/.moorlight/server.json")


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False


def viewer_url(cli=None):
    """URL of the running viewer. Raises RuntimeError if the state file points at a process that is gone."""
    if cli:
        return cli.rstrip("/")
    if os.environ.get("MOORLIGHT_URL"):
        return os.environ["MOORLIGHT_URL"].rstrip("/")
    try:
        with open(STATE_FILE) as fh:
            st = json.load(fh)
    except Exception:
        return "http://127.0.0.1:8765"
    if st.get("pid") and not _alive(st["pid"]):
        raise RuntimeError(f"{STATE_FILE} points at {st['url']} (pid {st['pid']}), but that viewer is not running; "
                           "start one with `moorlight ...` or pass --url")
    return st["url"].rstrip("/")


def http(url, path, body=None, timeout=600):
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def call(name, args=None, url=None, source="cli"):
    """Run one viewer command; returns the result or raises RuntimeError with the viewer's error message."""
    r = http(viewer_url(url), "/api/cmd", {"cmd": name, "args": args or {}, "source": source})
    if not r.get("ok"):
        raise RuntimeError(r.get("error", "error"))
    return r["result"]


def _value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def parse_args(tokens):
    """['chain=A', 'resseq=85'] or ['{"chain": "A"}'] -> dict."""
    out = {}
    for t in tokens:
        if t.startswith("{"):
            out.update(json.loads(t))
        elif "=" in t:
            k, v = t.split("=", 1)
            out[k] = _value(v)
        else:
            raise SystemExit(f"argument {t!r}: use key=value or a JSON object")
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    url = None
    if "--url" in argv:
        i = argv.index("--url")
        url = argv[i + 1]
        del argv[i:i + 2]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    try:
        base = viewer_url(url)
        if argv[0] == "--list":
            for c in http(base, "/api/commands"):
                if len(argv) > 1 and c["name"] not in argv[1:]:
                    continue
                print(f"{c['name']}: {c['description']}")
                for k, p in c["params"].items():
                    print(f"    {k} ({p['type']}): {p['description']}")
            return 0
        args = parse_args(argv[1:])
        for k in ("path", "file"):      # paths are relative to the caller, not to the viewer's working directory
            v = args.get(k)
            if isinstance(v, str) and v and not os.path.isabs(os.path.expanduser(v)) and os.path.exists(v):
                args[k] = os.path.abspath(v)
        result = call(argv[0], args, base)
    except (RuntimeError, urllib.error.URLError, ConnectionError) as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=1, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
