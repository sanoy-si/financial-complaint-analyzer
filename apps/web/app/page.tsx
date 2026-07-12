import Link from "next/link";
import { Logo } from "@/components/Logo";
import { FadeIn, Stagger, StaggerItem, SpringCard, Counter } from "@/components/MotionWrap";
import { TypingDemo } from "@/components/TypingDemo";

function Icon({ path }: { path: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round">
      <path d={path} />
    </svg>
  );
}

const ICONS = {
  upload: "M12 16V4m0 0L8 8m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2",
  chat:   "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",
  embed:  "M16 18l6-6-6-6M8 6l-6 6 6 6",
  shield: "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z",
  bolt:   "M13 2 3 14h9l-1 8 10-12h-9l1-8z",
  lock:   "M5 11h14v10H5zM8 11V7a4 4 0 0 1 8 0v4",
  star:   "M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z",
};

const STATS = [
  { num: 10000, suffix: "+", label: "Documents indexed" },
  { num: 99,    suffix: "%", label: "Grounded accuracy" },
  { num: 1,     suffix: "",  label: "Line of code to embed" },
  { num: 0,     suffix: "",  label: "Hallucinations, by design" },
];

const HOW = [
  {
    icon: ICONS.upload, step: "01", title: "Ingest",
    body: "Drop in a PDF or a URL. We extract, chunk, and embed it into a private vector store — one per project, fully isolated.",
  },
  {
    icon: ICONS.chat, step: "02", title: "Chat",
    body: "Ask in plain English. Answers are grounded strictly in your content and come back with the exact source passage.",
  },
  {
    icon: ICONS.embed, step: "03", title: "Embed",
    body: "Copy one <script> tag and a chat bubble appears on your own site, scoped entirely to your project key.",
  },
];

const FEATURES = [
  { icon: ICONS.shield, title: "Grounded & cited",   body: "No hallucinated answers — every response is backed by retrieved passages from your sources." },
  { icon: ICONS.bolt,   title: "No vendor lock-in",  body: "Local embeddings by default; swap the LLM (OpenAI, Groq…) with a single env var." },
  { icon: ICONS.lock,   title: "Private by design",  body: "Per-project isolation, domain allowlisting, and per-minute rate limits on the public widget API." },
];

const USE_CASES = [
  { title: "Complaint analysis",    body: "Spot emerging issues across thousands of narratives instantly." },
  { title: "Policy & handbook Q&A", body: "Answer staff questions from internal policy PDFs, cited to the page." },
  { title: "Product docs assistant",body: "A support bot grounded in your own documentation." },
];

export default function LandingPage() {
  return (
    <main style={{ background: "var(--bg)", color: "var(--ink)" }}>

      {/* NAV */}
      <nav className="nav">
        <div className="nav-inner">
          <div className="brand"><Logo /> Grounded</div>
          <div className="row">
            <Link className="btn btn-ghost" href="/login">Log in</Link>
            <Link className="btn btn-primary" href="/signup">Get started →</Link>
          </div>
        </div>
      </nav>

      {/* HERO */}
      <header className="hero">
        <div className="hero-bg">
          <div className="hero-blob hero-blob-1" />
          <div className="hero-blob hero-blob-2" />
          <div className="hero-blob hero-blob-3" />
        </div>
        <div className="hero-grid" />

        <div className="hero-inner">
          <div className="row" style={{ flexWrap: "wrap", gap: 48, alignItems: "center" }}>

            {/* copy */}
            <div style={{ flex: "1 1 460px" }}>
              <FadeIn>
                <span className="eyebrow">✦ Retrieval-augmented · grounded in your sources</span>
              </FadeIn>
              <FadeIn delay={0.07}>
                <h1 style={{ marginTop: 24 }}>
                  Turn your documents into a chatbot your customers can{" "}
                  <span className="gradient-text">actually&nbsp;trust</span>.
                </h1>
              </FadeIn>
              <FadeIn delay={0.14}>
                <p className="lead" style={{ marginTop: 22 }}>
                  Upload a PDF or paste a URL. Grounded ingests it, answers only
                  from your sources with citations, and hands you an embeddable
                  widget in one line of code.
                </p>
              </FadeIn>
              <FadeIn delay={0.21}>
                <div className="hero-cta row" style={{ marginTop: 36, flexWrap: "wrap" }}>
                  <Link className="btn btn-primary btn-lg" href="/signup">
                    Create your first bot →
                  </Link>
                  <Link className="btn btn-ghost btn-lg" href="/demo">
                    See live demo
                  </Link>
                </div>
                <p className="hero-note">Embeddings stay local · no credit card · open source</p>
              </FadeIn>
            </div>

            {/* live demo */}
            <FadeIn delay={0.3} className="fade-up" style={{ flex: "1 1 340px", maxWidth: 460 } as React.CSSProperties}>
              <TypingDemo />
            </FadeIn>
          </div>
        </div>
      </header>

      {/* STATS */}
      <div className="stats-strip">
        <div className="container">
          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(170px,1fr))", gap: 32 }}>
            {STATS.map((s) => (
              <FadeIn key={s.label}>
                <div style={{ textAlign: "center" }}>
                  <div className="stat-num">
                    <Counter to={s.num} suffix={s.suffix} />
                  </div>
                  <div className="stat-label">{s.label}</div>
                </div>
              </FadeIn>
            ))}
          </div>
        </div>
      </div>

      <div className="divider" />

      {/* HOW IT WORKS */}
      <section className="section">
        <div className="container">
          <FadeIn>
            <div className="center" style={{ marginBottom: 56 }}>
              <span className="eyebrow">How it works</span>
              <h2 style={{ fontSize: "clamp(28px,4vw,44px)", marginTop: 16 }}>
                From document to deployed bot<br />in minutes
              </h2>
            </div>
          </FadeIn>
          <Stagger className="grid grid-3">
            {HOW.map((h) => (
              <StaggerItem key={h.step}>
                <SpringCard>
                  <div className="card card-hover feature" style={{ height: "100%" }}>
                    <div className="row" style={{ marginBottom: 16, alignItems: "flex-start" }}>
                      <div className="icon-chip" style={{ marginBottom: 0 }}><Icon path={h.icon} /></div>
                      <span className="pill" style={{ marginLeft: "auto" }}>{h.step}</span>
                    </div>
                    <h3 style={{ marginTop: 12 }}>{h.title}</h3>
                    <p>{h.body}</p>
                  </div>
                </SpringCard>
              </StaggerItem>
            ))}
          </Stagger>
        </div>
      </section>

      <div className="divider" />

      {/* FEATURES */}
      <section className="section" style={{ background: "var(--bg-2)" }}>
        <div className="container">
          <FadeIn>
            <div className="center" style={{ marginBottom: 52 }}>
              <span className="eyebrow">Why Grounded</span>
              <h2 style={{ fontSize: "clamp(28px,4vw,40px)", marginTop: 16 }}>Built differently</h2>
            </div>
          </FadeIn>
          <Stagger className="grid grid-3">
            {FEATURES.map((f) => (
              <StaggerItem key={f.title}>
                <SpringCard>
                  <div className="card card-hover feature" style={{ height: "100%" }}>
                    <div className="icon-chip"><Icon path={f.icon} /></div>
                    <h3>{f.title}</h3>
                    <p>{f.body}</p>
                  </div>
                </SpringCard>
              </StaggerItem>
            ))}
          </Stagger>
        </div>
      </section>

      <div className="divider" />

      {/* USE CASES */}
      <section className="section">
        <div className="container">
          <FadeIn>
            <div className="center" style={{ marginBottom: 52 }}>
              <span className="eyebrow">Built for finance &amp; compliance</span>
              <h2 style={{ fontSize: "clamp(28px,4vw,40px)", marginTop: 16 }}>Your higher-value use cases</h2>
              <p className="muted" style={{ maxWidth: 560, margin: "14px auto 0", fontSize: 17 }}>
                Make sense of complaints, policies, and filings. The engine stays fully generic.
              </p>
            </div>
          </FadeIn>
          <Stagger className="grid grid-3">
            {USE_CASES.map((u) => (
              <StaggerItem key={u.title}>
                <SpringCard>
                  <div className="card card-hover" style={{ height: "100%" }}>
                    <div className="row" style={{ marginBottom: 14 }}>
                      <div className="icon-chip" style={{ marginBottom: 0 }}><Icon path={ICONS.star} /></div>
                      <span className="pill">sample dataset</span>
                    </div>
                    <strong style={{ fontSize: 16, color: "var(--ink)" }}>{u.title}</strong>
                    <p className="muted" style={{ marginTop: 8, fontSize: 14 }}>{u.body}</p>
                  </div>
                </SpringCard>
              </StaggerItem>
            ))}
          </Stagger>
          <FadeIn>
            <div className="center" style={{ marginTop: 48 }}>
              <Link className="btn btn-primary btn-lg" href="/signup">Start building free →</Link>
            </div>
          </FadeIn>
        </div>
      </section>

      {/* CTA BAND */}
      <section style={{ background: "var(--bg-2)", borderTop: "1px solid var(--line)", padding: "80px 0" }}>
        <div className="container">
          <FadeIn>
            <div className="glow-border" style={{
              background: "var(--panel)", backdropFilter: "blur(18px)",
              padding: "60px 40px", textAlign: "center",
              maxWidth: 720, margin: "0 auto", borderRadius: "var(--radius)",
            }}>
              <div className="icon-chip" style={{ margin: "0 auto 20px" }}><Icon path={ICONS.bolt} /></div>
              <h2 style={{ fontSize: "clamp(24px,3.5vw,38px)" }}>Ready to ground your chatbot?</h2>
              <p className="muted" style={{ margin: "14px auto 0", maxWidth: 480, fontSize: 16 }}>
                Create an account in 30 seconds. No credit card. Use the free mock mode to explore
                the product first, then plug in your own LLM key when ready.
              </p>
              <div className="row" style={{ justifyContent: "center", marginTop: 32, flexWrap: "wrap" }}>
                <Link className="btn btn-primary btn-lg" href="/signup">Create free account</Link>
                <Link className="btn btn-ghost btn-lg" href="/login">Log in</Link>
              </div>
            </div>
          </FadeIn>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="footer">
        <div className="container row" style={{ justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
          <div className="brand" style={{ fontSize: 16 }}><Logo /> Grounded</div>
          <span className="muted">A deterministic, no-lock-in RAG platform.</span>
        </div>
      </footer>

    </main>
  );
}
