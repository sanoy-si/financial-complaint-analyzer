"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  api,
  getToken,
  API_BASE,
  type Project,
  type DocumentItem,
  type Source,
} from "@/lib/api";

type Tab = "content" | "chat" | "embed";
interface ChatMsg {
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
}

export default function ProjectPage() {
  const router = useRouter();
  const params = useParams<{ id: string }>();
  const projectId = params.id;

  const [project, setProject] = useState<Project | null>(null);
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [tab, setTab] = useState<Tab>("content");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!getToken()) {
      router.push("/login");
      return;
    }
    api.getProject(projectId).then(setProject).catch((e) => setError(e.message));
    api.listDocuments(projectId).then(setDocs).catch(() => undefined);
  }, [projectId, router]);

  async function refreshDocs() {
    setDocs(await api.listDocuments(projectId));
  }

  return (
    <main>
      <nav className="nav">
        <Link href="/dashboard"><strong>Grounded</strong></Link>
        {project && <span className="muted">{project.name}</span>}
      </nav>
      <div className="container">
        {error && <p className="error">{error}</p>}
        <div className="tabs">
          {(["content", "chat", "embed"] as Tab[]).map((t) => (
            <div key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
              {t === "content" ? "Content" : t === "chat" ? "Chat" : "Embed"}
            </div>
          ))}
        </div>

        {tab === "content" && (
          <ContentTab projectId={projectId} docs={docs} onChange={refreshDocs} />
        )}
        {tab === "chat" && <ChatTab projectId={projectId} />}
        {tab === "embed" && project && <EmbedTab project={project} />}
      </div>
    </main>
  );
}

function ContentTab({
  projectId,
  docs,
  onChange,
}: {
  projectId: string;
  docs: DocumentItem[];
  onChange: () => void;
}) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  async function addUrl(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim()) return;
    setBusy(true);
    setError("");
    try {
      await api.ingestUrl(projectId, url.trim());
      setUrl("");
      onChange();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed");
    } finally {
      setBusy(false);
    }
  }

  async function addPdf() {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      await api.uploadPdf(projectId, file);
      if (fileRef.current) fileRef.current.value = "";
      onChange();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Add content</h3>
        <form onSubmit={addUrl} className="row">
          <input className="input" placeholder="https://example.com/page" value={url}
            onChange={(e) => setUrl(e.target.value)} />
          <button className="btn" type="submit" disabled={busy}>Add URL</button>
        </form>
        <div className="row" style={{ marginTop: 12 }}>
          <input ref={fileRef} type="file" accept="application/pdf" />
          <button className="btn secondary" onClick={addPdf} disabled={busy}>Upload PDF</button>
        </div>
        {error && <p className="error">{error}</p>}
      </div>

      <h3>Indexed documents</h3>
      {docs.length === 0 ? (
        <p className="muted">No documents yet.</p>
      ) : (
        <ul>
          {docs.map((d) => (
            <li key={d.id}>
              <span className="tag">{d.source_type}</span> {d.source_ref}{" "}
              <span className="muted">— {d.status}</span>
              {d.error && <span className="error"> ({d.error})</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ChatTab({ projectId }: { projectId: string }) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const sessionId = useMemo(
    () => "play_" + Math.random().toString(36).slice(2, 10),
    []
  );

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    const question = q.trim();
    setMessages((m) => [...m, { role: "user", text: question }]);
    setQ("");
    setBusy(true);
    try {
      const res = await api.chat(projectId, question, sessionId);
      setMessages((m) => [
        ...m,
        { role: "assistant", text: res.answer, sources: res.sources },
      ]);
    } catch {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: "Something went wrong. Try again." },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="card" style={{ minHeight: 280, marginBottom: 12 }}>
        {messages.length === 0 ? (
          <p className="muted">Ask a question grounded in this project&apos;s content.</p>
        ) : (
          messages.map((m, i) => (
            <div key={i} style={{ margin: "8px 0" }}>
              <strong>{m.role === "user" ? "You" : "Bot"}:</strong> {m.text}
              {m.sources && m.sources.length > 0 && (
                <div className="muted" style={{ fontSize: 12 }}>
                  {m.sources.length} source passage(s)
                </div>
              )}
            </div>
          ))
        )}
      </div>
      <form onSubmit={ask} className="row">
        <input className="input" placeholder="Type a question…" value={q}
          onChange={(e) => setQ(e.target.value)} />
        <button className="btn" type="submit" disabled={busy}>Send</button>
      </form>
    </div>
  );
}

function EmbedTab({ project }: { project: Project }) {
  const snippet = `<script src="${API_BASE}/widget.js"
        data-project-key="${project.public_key}"
        data-api-base="${API_BASE}"></script>`;
  return (
    <div>
      <h3>Embed on your site</h3>
      <p className="muted">
        Paste this snippet just before the closing &lt;/body&gt; tag on any page.
      </p>
      <pre>{snippet}</pre>
      <button className="btn secondary" onClick={() => navigator.clipboard.writeText(snippet)}>
        Copy snippet
      </button>
    </div>
  );
}
