# Agent Server Integration

This service can run as the algorithm backend for `agent-server`.

## Runtime Configuration

Set these environment variables before starting the Flask app:

```bash
ALGORITHM_API_KEY=<shared-secret>
ALGORITHM_HOST=0.0.0.0
ALGORITHM_PORT=9000
ALGORITHM_DEFAULT_CHARACTER_ID=emi
```

`ALGORITHM_API_KEY` is required for `/stream`.

## Start

```bash
uv sync --frozen
ALGORITHM_API_KEY=<shared-secret> ALGORITHM_PORT=9000 uv run python app.py
```

On Windows PowerShell:

```powershell
$env:ALGORITHM_API_KEY="<shared-secret>"
$env:ALGORITHM_PORT="9000"
uv run python app.py
```

## Stream API

`agent-server` calls:

```text
POST /stream
Authorization: Bearer <shared-secret>
Content-Type: application/json
```

Request body:

```json
{
  "device_id": "device-1",
  "session_id": "session-1",
  "user_text": "hello",
  "conversation_context": [],
  "session_state": ""
}
```

Response body is SSE-compatible:

```text
data: {"delta":"...","request_id":"..."}
data: {"delta":"...","request_id":"..."}
data: [DONE]
```

## Smoke Test

```bash
curl -N http://127.0.0.1:9000/stream \
  -H "Authorization: Bearer <shared-secret>" \
  -H "Content-Type: application/json" \
  -d '{"device_id":"test","session_id":"test","user_text":"hello","conversation_context":[],"session_state":""}'
```

## agent-server `.env`

```env
RESPONSE_ENGINE_PROVIDER=algorithm
ALGORITHM_BASE_URL=http://127.0.0.1:9000
ALGORITHM_API_KEY=<shared-secret>
ALGORITHM_STREAMING_ENABLED=true
```

## Debugging

Each `/stream` request logs:

- `request_id`
- `device_id`
- `session_id`
- input length
- first delta latency
- total completion latency
- token event count

The logs intentionally avoid printing full user input or full assistant replies.
