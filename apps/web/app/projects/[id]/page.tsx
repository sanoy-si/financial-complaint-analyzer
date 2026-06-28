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
    <div className="app-bg">
      <nav className="nav">
        <div className="nav-inner">
          <Link href="/dashboard" className="brand"><span className="logo" /> Grounded</Link>
          {project && <span className="pill">{project.name}</span>}
        </div>
      </nav>

      <div className="container" style={{ paddingTop: 32, paddingBottom: 60 }}>
        <Link href="/dashboard" className="muted" style={{ fontSize: 14 }}>← All chatbots</Link>
        <h1 style={{ fontSize: 30, margin: "8px 0 20px" }}>{project?.name ?? "Project"}</h1>
        {error && <p className="error">{error}</p>}

        <div className="tabs">
          {(["content", "chat", "embed"] as Tab[]).map((t) => (
            <div key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
              {t === "content" ? "Content" : t === "chat" ? "Chat" : "Embed"}
            </div>
          ))}
        </div>

        <div className="fade-up">
          {tab === "content" && <ContentTab projectId={projectId} docs={docs} onChange={refreshDocs} />}
          {tab === "chat" && <ChatTab projectId={projectId} />}
          {tab === "embed" && project && <EmbedTab project={project} />}
        </div>
      </div>
    </div>
  );
}

function ContentTab({ projectId, docs, onChange }: {
  projectId: string; docs: DocumentItem[]; onChange: () => void;
}) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  async function addUrl(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim()) return;
    setBusy(true); setError("");
    try { await api.ingestUrl(projectId, url.trim()); setUrl(""); onChange(); }
    catch (e) { setError(e instanceof Error ? e.message : "Failed"); }
    finally { setBusy(false); }
  }

  async function addPdf() {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setBusy(true); setError("");
    try { await api.uploadPdf(projectId, file); if (fileRef.current) fileRef.current.value = ""; onChange(); }
    catch (e) { setError(e instanceof Error ? e.message : "Upload failed"); }
    finally { setBusy(false); }
  }

  return (
    <div className="grid" style={{ gridTemplateColumns: "1fr", gap: 24 }}>
      <div className="card">
        <h3 style={{ fontSize: 18 }}>Add content</h3>
        <p className="muted" style={{ marginTop: 4, fontSize: 14 }}>Paste a public URL or upload a PDF. We index it instantly.</p>
        <form onSubmit={addUrl} className="row" style={{ marginTop: 14 }}>
          <input className="input" placeholder="https://example.com/page" value={url} onChange={(e) => setUrl(e.target.value)} />
          <button className="btn btn-primary" type="submit" disabled={busy}>Add URL</button>
        </form>
        <div className="row" style={{ marginTop: 12 }}>
          <input ref={fileRef} type="file" accept="application/pdf" style={{ fontSize: 14 }} />
          <button className="btn btn-ghost" onClick={addPdf} disabled={busy}>Upload PDF</button>
        </div>
        {busy && <p className="muted" style={{ marginTop: 10 }}>Indexing…</p>}
        {error && <p className="error">{error}</p>}
      </div>

      <div>
        <h3 style={{ fontSize: 18, marginBottom: 12 }}>Indexed documents</h3>
        {docs.length === 0 ? (
          <div className="empty">Nothing indexed yet — add a URL or PDF above.</div>
        ) : (
          docs.map((d) => (
            <div key={d.id} className="doc-item">
              <span className={`dot ${d.status}`} />
              <span className="pill">{d.source_type}</span>
              <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{d.source_ref}</span>
              <span className="muted" style={{ fontSize: 13 }}>{d.status}</span>
              {d.error && <span className="error" style={{ fontSize: 12 }}>· {d.error}</span>}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function ChatTab({ projectId }: { projectId: string }) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const sessionId = useMemo(() => "play_" + Math.random().toString(36).slice(2, 10), []);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    const question = q.trim();
    setMessages((m) => [...m, { role: "user", text: question }]);
    setQ(""); setBusy(true);
    try {
      const res = await api.chat(projectId, question, sessionId);
      setMessages((m) => [...m, { role: "assistant", text: res.answer, sources: res.sources }]);
    } catch {
      setMessages((m) => [...m, { role: "assistant", text: "Something went wrong. Try again." }]);
    } finally { setBusy(false); }
  }

  return (
    <div>
      <div className="chat-log">
        {messages.length === 0 ? (
          <p className="muted" style={{ margin: "auto" }}>Ask anything grounded in this project&apos;s content.</p>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              {m.text}
              {m.sources && m.sources.length > 0 && (
                <div className="src">📎 {m.sources.length} source passage(s)</div>
              )}
            </div>
          ))
        )}
        {busy && <div className="msg bot">…</div>}
      </div>
      <form onSubmit={ask} className="composer">
        <input className="input" placeholder="Type a question…" value={q} onChange={(e) => setQ(e.target.value)} />
        <button className="btn btn-primary" type="submit" disabled={busy}>Send</button>
      </form>
    </div>
  );
}

function EmbedTab({ project }: { project: Project }) {
  const [copied, setCopied] = useState(false);
  const snippet = `<script src="${API_BASE}/widget.js"
        data-project-key="${project.public_key}"
        data-api-base="${API_BASE}"></script>`;

  function copy() {
    navigator.clipboard.writeText(snippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="card">
      <h3 style={{ fontSize: 18 }}>Embed on your site</h3>
      <p className="muted" style={{ marginTop: 4, fontSize: 14 }}>
        Paste this just before the closing <code>&lt;/body&gt;</code> tag on any page. The chat
        bubble appears in the corner, scoped to this project.
      </p>
      <pre className="pre" style={{ marginTop: 14 }}>{snippet}</pre>
      <button className="btn btn-primary" onClick={copy} style={{ marginTop: 12 }}>
        {copied ? "Copied ✓" : "Copy snippet"}
      </button>
    </div>
  );
}
