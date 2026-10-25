# TDS GA3 — A2A Invoice Agent

This is a starter implementation for the A2A Invoice Agent assignment. It includes a FastAPI service, SQLite persistence, per-token task isolation, message idempotency, proposal artifacts, result continuation handling, and a deploy manifest.

## Important limitations before submitting
- This is a starter, not a guarantee of full marks. The assignment's package-specific evidence and action logic is difficult; test with the actual grader before using the final Save.
- SQLite on a free ephemeral host may lose data after restart/redeploy. Use a persistent disk or hosted Postgres for production-grade persistence; the code currently uses SQLite.
- Every syntactically valid Bearer token is treated as a separate principal, so task data is isolated by token. Follow the grader's authentication instructions if the assignment page specifies a required token. Do not share tokens in chat or commit secrets to Git.
- Configure `BASE_URL` to the exact submitted base URL, including `/a2a/`.
- Set `MODEL_API_KEY` to a compatible OpenAI API key if using a hosted model. Without a key, conservative fallback decisions are used and accuracy will be limited.
- The current starter's rule-based fallback is intentionally conservative; it does not replace evaluation against the hidden audit batches.
- Never manually fabricate grader receipt nonces or execute rejected proposals.

## Local run (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:API_TOKEN="local-test-token"
$env:BASE_URL="http://127.0.0.1:8000/a2a/"
uvicorn main:app --reload
```
Local HTTP is only for development. Submission requires public HTTPS.

## Deploy on Render (manual steps)
1. Create a GitHub repository and upload these project files.
2. In Render, create a new Web Service from the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Set environment variables in Render: `API_TOKEN`, `BASE_URL`, `MODEL_API_KEY`, `MODEL_NAME`, `DB_PATH`.
6. Ensure a persistent disk is configured for SQLite at a persistent path, or adapt the DB to a persistent managed database.
7. After deploy, verify `https://YOUR-SERVICE.onrender.com/health` and `https://YOUR-SERVICE.onrender.com/.well-known/agent-card.json`.
8. Submit `https://YOUR-SERVICE.onrender.com/a2a/` only after the agent card's `supportedInterfaces[0].url` exactly matches it and the grader token/auth contract is configured.

## Routes
- `GET /.well-known/agent-card.json` — public card
- `POST /a2a/message:send`
- `GET /a2a/tasks`
- `GET /a2a/tasks/{task_id}`
- `POST /a2a/tasks/{task_id}:cancel`

All non-card routes require `Authorization: Bearer <API_TOKEN>`, `A2A-Version: 1.0`, and `Content-Type: application/a2a+json`.

## Model
The app sends one batched request for uncached packages to an OpenAI-compatible `/chat/completions` endpoint. Configure `MODEL_API_KEY`, `MODEL_BASE_URL`, and `MODEL_NAME` in the host. Treat invoice documents as untrusted evidence.
