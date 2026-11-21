"""Readable transcript of the OpenAI-compatible surface, for a human.

Run it against whatever engine is listening:

    ./bin/qe-serve --port 8099 --threads 20 --id qe-selftest &
    python3 tools/demo_http.py

    ./bin/qe-serve --port 8080 --model .../-00001-of-00003.gguf &
    QE_PORT=8080 python3 tools/demo_http.py

It asserts nothing. tools/smoke_http.py and tools/fuzz_http.py are the tests;
this one exists so that "what does it actually answer" is one command away.
"""
import base64
import http.client
import json
import os
import sys

PORT = int(os.environ.get("QE_PORT", "8099"))
PATH = os.environ.get("QE_MODEL_PATH", "")

# 1x1 PNG. The engine does not run a vision encoder, so this exercises the
# request path and the image accounting, not any pixels.
PNG = base64.b64encode(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
    )
).decode()


def req(method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    c.request(
        method,
        path,
        body=json.dumps(body) if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    r = c.getresponse()
    data = r.read().decode("utf-8", "replace")
    c.close()
    return r.status, data


def show(title, status, body, keys=None):
    print(f"\n=== {title} -> HTTP {status}")
    try:
        j = json.loads(body)
    except ValueError:
        print(body[:400])
        return None
    print(json.dumps({k: j.get(k) for k in keys} if keys and isinstance(j, dict) else j, indent=2)[:1600])
    return j


print(f"engine on 127.0.0.1:{PORT}")

st, body = req("GET", "/health")
show("GET /health", st, body)

st, body = req("GET", "/v1/models")
models = show("GET /v1/models", st, body, keys=["object"])
m0 = json.loads(body)["data"][0]
model = m0["id"]
qe = m0.get("qe")
if qe is None:
    print("    WARNING: no 'qe' block on /v1/models, so this is not qe-serve."
          " The port is probably taken by something else.")
    sys.exit(2)
print(f"    served id: {model}  can_generate: {qe['can_generate']}"
      f"  mtp_head: {qe['mtp_head']}  tensors: {qe['tensors']}")

st, body = req(
    "POST",
    "/v1/chat/completions",
    {
        "model": model,
        "messages": [{"role": "user", "content": "Explain why decode is bandwidth-bound."}],
        "max_tokens": 24,
        "temperature": 0.0,
    },
)
chat = show("POST /v1/chat/completions (non-stream)", st, body, keys=["object", "choices", "usage"])
if chat and chat.get("error"):
    show("     error body", st, json.dumps(chat["error"], indent=2))
elif chat and chat.get("choices"):
    print("    content:", chat["choices"][0]["message"]["content"][:160])
    print("    qe block:", json.dumps(chat.get("qe", {}))[:300])

st, body = req(
    "POST",
    "/v1/chat/completions",
    {
        "model": model,
        "stream": True,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "stream a sentence"}],
    },
)
print(f"\n=== POST /v1/chat/completions (stream: true) -> HTTP {st}")
lines = [ln for ln in body.splitlines() if ln.startswith("data:")]
for ln in lines[:3]:
    print("   ", ln[:120])
print(f"    ... {len(lines)} data lines total, last: {lines[-1][:60] if lines else 'none'}")

st, body = req(
    "POST",
    "/v1/chat/completions",
    {
        "model": model,
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
                ],
            }
        ],
    },
)
img = show("POST /v1/chat/completions (text + base64 image)", st, body, keys=["object", "error"])
if img and not img.get("error"):
    print("    qe block:", json.dumps(img.get("qe", {}))[:300])

st, body = req("POST", "/v1/completions", {"model": model, "prompt": ["alpha", "beta"], "max_tokens": 8})
show("POST /v1/completions (array prompt)", st, body, keys=["object", "error"])

if PATH:
    st, body = req(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
    )
    err = json.loads(body).get("error", {})
    print(f"\n=== real checkpoint generate -> HTTP {st}")
    print("    code:", err.get("code"), " type:", err.get("type"))
    why = (err.get("info") or {}).get("why") or err.get("message", "")
    print("    why:", why[:500])

print()
sys.exit(0)
