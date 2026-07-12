"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "motion/react";
import { Logo } from "@/components/Logo";
import { FadeIn } from "@/components/MotionWrap";
import { api, type DemoBot, type Source } from "@/lib/api";

interface ChatMsg {
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
}

export default function DemoPage() {
  const [demos, setDemos] = useState<DemoBot[] | null>(null);
  const [error, setError] = useState("");
  const [activeKey, setActiveKey] = useState<string>("");

  useEffect(() => {
    api
      .listDemos()
      .then((list) => {
        setDemos(list);
        if (list.length > 0) setActiveKey(list[0].public_key);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load demos"));
  }, []);

  const active = demos?.find((d) => d.public_key === activeKey) ?? null;

  return (
    <div className="app-bg">
      <nav className="nav">
        <div className="nav-inner">
          <Link href="/" className="brand"><Logo /> Grounded</Link>
          <div className="row">
            <Link className="btn btn-ghost" href="/">← Home</Link>
            <Link className="btn btn-primary" href="/signup">Get started →</Link>
          </div>
        </div>
      </nav>

      <div className="container" style={{ paddingTop: 40, paddingBottom: 80, maxWidth: 760 }}>
        <FadeIn>
          <span className="eyebrow">✦ Live · no login required</span>
          <h1 style={{ fontSize: "clamp(28px,4vw,40px)", margin: "16px 0 10px" }}>
            Try it on a real dataset
          </h1>
          <p className="muted" style={{ fontSize: 16, maxWidth: 560 }}>
            Pick a sample knowledge base and ask a question. Every answer is grounded
            strictly in that dataset and comes back with the source passage — the exact
            behaviour you get on your own documents.
          </p>
        </FadeIn>

        {error && (
          <p className="error" style={{ marginTop: 24 }}>{error}</p>
        )}

        {demos === null && !error && (
          <p className="muted" style={{ marginTop: 32 }}>Loading demos…</p>
        )}

        {demos !== null && demos.length === 0 && !error && (
          <div className="empty" style={{ marginTop: 32 }}>
            No demo datasets are available right now.{" "}
            <Link href="/signup" style={{ color: "var(--accent, #a78bfa)" }}>
              Create a free account
            </Link>{" "}
            to build your own.
          </div>
        )}

        {active && (
          <div style={{ marginTop: 28 }}>
            <div className="tabs">
              {demos!.map((d) => (
                <div
                  key={d.public_key}
                  className={`tab ${d.public_key === activeKey ? "active" : ""}`}
                  onClick={() => setActiveKey(d.public_key)}
                >
                  {d.name}
                </div>
              ))}
            </div>
            {/* Remount the chat when the dataset changes so history/session reset. */}
            <DemoChat key={active.public_key} bot={active} />
          </div>
        )}
      </div>
    </div>
  );
}

function DemoChat({ bot }: { bot: DemoBot }) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);
  const sessionId = useMemo(() => "demo_" + Math.random().toString(36).slice(2, 10), []);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  async function send(question: string) {
    const text = question.trim();
    if (!text || busy) return;
    setMessages((m) => [...m, { role: "user", text }]);
    setQ("");
    setBusy(true);
    try {
      const res = await api.publicChat(bot.public_key, text, sessionId);
      setMessages((m) => [...m, { role: "assistant", text: res.answer, sources: res.sources }]);
    } catch {
      setMessages((m) => [...m, { role: "assistant", text: "Something went wrong. Try again." }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ marginTop: 20 }}>
      {messages.length === 0 && bot.sample_questions.length > 0 && (
        <div className="row" style={{ flexWrap: "wrap", gap: 8, marginBottom: 14 }}>
          {bot.sample_questions.map((sq) => (
            <button
              key={sq}
              className="pill"
              onClick={() => send(sq)}
              style={{ cursor: "pointer", border: "none" }}
            >
              {sq}
            </button>
          ))}
        </div>
      )}

      <div className="chat-log" ref={logRef}>
        {messages.length === 0 ? (
          <p className="muted" style={{ margin: "auto", textAlign: "center", fontSize: 14 }}>
            Ask a question, or tap a suggestion above.
          </p>
        ) : (
          <AnimatePresence initial={false}>
            {messages.map((m, i) => (
              <motion.div
                key={i}
                className={`msg ${m.role}`}
                initial={{ opacity: 0, y: 10, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ type: "spring", stiffness: 300, damping: 28 }}
              >
                {m.text}
                {m.sources && m.sources.length > 0 && (
                  <div className="src">📎 {m.sources.length} source passage(s)</div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
        )}
        {busy && (
          <motion.div
            className="msg bot"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{ fontSize: 20, letterSpacing: 6 }}
          >
            ···
          </motion.div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(q);
        }}
        className="composer"
      >
        <input
          className="input"
          placeholder="Type a question…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          style={{ flex: 1 }}
        />
        <button className="btn btn-primary" type="submit" disabled={busy}>Send</button>
      </form>
    </div>
  );
}
