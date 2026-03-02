# Grounded embeddable widget

A single, dependency-free script that drops a grounded chat bubble onto any
website. It renders inside a Shadow DOM (host-page styles can't leak in) and
talks to the public, key-scoped chat API.

## Embed

```html
<script src="https://app.example.com/widget.js"
        data-project-key="pk_your_project_key"
        data-api-base="https://api.example.com"
        data-title="Ask about our docs"></script>
```

| Attribute | Required | Description |
|-----------|----------|-------------|
| `data-project-key` | yes | The project's public key (`pk_…`) from the dashboard. |
| `data-api-base` | no | API origin. Defaults to the page origin. |
| `data-title` | no | Header text on the chat panel. |

## Security

- The public key is **not a secret** — it only allows asking questions against
  that one project's indexed content.
- Lock a project to specific domains in its settings (`allowed_domains`); requests
  with a disallowed `Origin` are rejected.
- The public endpoint is rate-limited per key.

## Local demo

Serve this folder statically and open `demo.html` (set `data-project-key` to a
real key from your local dashboard, and `data-api-base` to `http://localhost:8000`).
