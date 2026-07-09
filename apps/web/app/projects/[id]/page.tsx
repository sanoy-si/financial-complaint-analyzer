"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { motion, AnimatePresence } from "motion/react";
import { Logo } from "@/components/Logo";
import { FadeIn } from "@/components/MotionWrap";
import {
  api, getToken, API_BASE,
  type Project, type DocumentItem, type Source,
} from "@/lib/api";

type Tab = "content" | "chat" | "embed";
interface ChatMsg { role: "user" | "assistant"; text: string; sources?: Source[]; }

export default function ProjectPage() {
  const router = useRouter();
  const params = useParams<{ id: string }>();
  const projectId = params.id;

  const [project, setProject] = useState<Project | null>(null);
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [tab, setTab] = useState<Tab>("content");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!getToken()) { router.push("/login"); return; }
    api.getProject(projectId).then(setProject).catch((e) => setError(e.message));
    api.listDocuments(projectId).then(setDocs).catch(() => undefined);
  }, [projectId, router]);

  async function refreshDocs() { setDocs(await api.listDocuments(projectId)); }

  const TABS: { key: Tab; label: string }[] = [
    { key: "content", label: "Content" },
    { key: "chat",    label: "Chat" },
    { key: "embed",   label: "Embed" },
  ];

  return (
    <div className="app-bg">
      <nav className="nav">
        <div className="nav-inner">
          <Link href="/dashboard" className="brand"><Logo /> Grounded</Link>
          {project && <span className="pill">{project.name}</span>}
        </div>
      </nav>

      <div className="container" style={{ paddingTop: 36, paddingBottom: 72 }}>
        <FadeIn>
          <Link href="/dashboard" className="muted" style={{ fontSize: 13 }}>← All chatbots</Link>
          <h1 style={{ fontSize: 30, margin: "10px 0 24px" }}>{project?.name ?? "Project"}</h1>
          {error && <p className="error">{error}</p>}

          <div className="tabs">
            {TABS.map((t) => (
              <div key={t.key} className={`tab ${tab === t.key ? "active" : ""}`}
                onClick={() => setTab(t.key)}>
                {t.label}
              </div>
            ))}
          </div>
        </FadeIn>

        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.22 }}
          >
            {tab === "content" && <ContentTab projectId={projectId} docs={docs} onChange={refreshDocs} />}
            {tab === "chat"    && <ChatTab projectId={projectId} />}
            {tab === "embed"   && project && <EmbedTab project={project} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

/* ── CONTENT TAB ─────────────────────────────────── */
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
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <div className="card">
        <h3 style={{ fontSize: 18, marginBottom: 6 }}>Add content</h3>
        <p className="muted" style={{ fontSize: 14, marginBottom: 16 }}>Paste a public URL or upload a PDF. We index it instantly.</p>
        <form onSubmit={addUrl} className="row" style={{ marginBottom: 12 }}>
          <input className="input" placeholder="https://example.com/page"
            value={url} onChange={(e) => setUrl(e.target.value)} style={{ flex: 1 }} />
          <button className="btn btn-primary" type="submit" disabled={busy}>Add URL</button>
        </form>
        <div className="row">
          <input ref={fileRef} type="file" accept="application/pdf"
            style={{ fontSize: 14, color: "var(--muted)", flex: 1 }} />
          <button className="btn btn-ghost" onClick={addPdf} disabled={busy}>Upload PDF</button>
        </div>
        {busy && <p className="muted" style={{ marginTop: 12, fontSize: 14 }}>Indexing…</p>}
        {error && <p className="error" style={{ marginTop: 8 }}>{error}</p>}
      </div>

      <div>
        <h3 style={{ fontSize: 18, marginBottom: 14 }}>Indexed documents</h3>
        {docs.length === 0 ? (
          <div className="empty">Nothing indexed yet — add a URL or PDF above.</div>
        ) : (
          <AnimatePresence>
            {docs.map((d) => (
              <motion.div key={d.id} className="doc-item"
                initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <span className={`dot ${d.status}`} />
                <span className="pill">{d.source_type}</span>
                <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: 14 }}>
                  {d.source_ref}
                </span>
                <span className="muted" style={{ fontSize: 13 }}>{d.status}</span>
                {d.error && <span className="error" style={{ fontSize: 12 }}>· {d.error}</span>}
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}

/* ── CHAT TAB ─────────────────────────────────────── */
function ChatTab({ projectId }: { projectId: string }) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);
  const sessionId = useMemo(() => "play_" + Math.random().toString(36).slice(2, 10), []);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

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
      <div className="chat-log" ref={logRef}>
        {messages.length === 0 ? (
          <p className="muted" style={{ margin: "auto", textAlign: "center", fontSize: 14 }}>
            Ask anything grounded in this project&apos;s content.
          </p>
        ) : (
          <AnimatePresence initial={false}>
            {messages.map((m, i) => (
              <motion.div key={i} className={`msg ${m.role}`}
                initial={{ opacity: 0, y: 10, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ type: "spring", stiffness: 300, damping: 28 }}>
                {m.text}
                {m.sources && m.sources.length > 0 && (
                  <div className="src">📎 {m.sources.length} source passage(s)</div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
        )}
        {busy && (
          <motion.div className="msg bot" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            style={{ fontSize: 20, letterSpacing: 6 }}>···</motion.div>
        )}
      </div>
      <form onSubmit={ask} className="composer">
        <input className="input" placeholder="Type a question…"
          value={q} onChange={(e) => setQ(e.target.value)} style={{ flex: 1 }} />
        <button className="btn btn-primary" type="submit" disabled={busy}>Send</button>
      </form>
    </div>
  );
}

/* ── EMBED TAB ────────────────────────────────────── */
function EmbedTab({ project }: { project: Project }) {
  const [copied, setCopied] = useState(false);
  const snippet = `<script src="${API_BASE}/widget.js"\n  data-project-key="${project.public_key}"\n  data-api-base="${API_BASE}"></script>`;

  function copy() {
    navigator.clipboard.writeText(snippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }

  return (
    <div className="card" style={{ maxWidth: 640 }}>
      <h3 style={{ fontSize: 18 }}>Embed on your site</h3>
      <p className="muted" style={{ marginTop: 6, fontSize: 14, lineHeight: 1.6 }}>
        Paste this just before the closing <code>&lt;/body&gt;</code> tag on any page.
        The chat bubble appears in the corner, scoped to this project.
      </p>
      <pre className="pre" style={{ marginTop: 18 }}>{snippet}</pre>
      <motion.button className="btn btn-primary" onClick={copy}
        style={{ marginTop: 14 }} whileTap={{ scale: 0.96 }}>
        {copied ? "✓ Copied!" : "Copy snippet"}
      </motion.button>
    </div>
  );
}
