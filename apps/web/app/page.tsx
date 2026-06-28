import Link from "next/link";
import { Logo } from "@/components/Logo";

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
  chat: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",
  embed: "M16 18l6-6-6-6M8 6l-6 6 6 6",
  shield: "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z",
  bolt: "M13 2 3 14h9l-1 8 10-12h-9l1-8z",
  lock: "M5 11h14v10H5zM8 11V7a4 4 0 0 1 8 0v4",
};

export default function LandingPage() {
  return (
    <main>
      <nav className="nav on-dark">
        <div className="nav-inner">
          <div className="brand"><Logo /> Grounded</div>
          <div className="row">
            <Link className="btn btn-ghost on-dark" href="/login">Log in</Link>
            <Link className="btn btn-primary" href="/signup">Get started</Link>
          </div>
        </div>
      </nav>

      <header className="hero">
        <div className="hero-bg" />
        <div className="hero-grid" />
        <div className="hero-inner">
          <span className="eyebrow fade-up">✦ Retrieval-augmented, grounded in your sources</span>
          <h1 className="fade-up d1">
            Turn your documents into a chatbot<br />
            your customers can <span className="gradient-text">actually trust</span>.
          </h1>
          <p className="lead fade-up d2">
            Upload a PDF or paste a URL. Grounded ingests the content, answers only
            from your sources with citations, and hands you an embeddable widget to
            drop on your site with one line of code.
          </p>
          <div className="hero-cta row fade-up d3">
            <Link className="btn btn-primary btn-lg" href="/signup">Create your first bot →</Link>
            <Link className="btn btn-ghost on-dark btn-lg" href="/login">Live demo</Link>
          </div>
          <p className="hero-note fade-up d3">
            Documents are embedded locally and never leave your server. No credit card.
          </p>
        </div>
      </header>

      <section className="section">
        <div className="container">
          <div className="center" style={{ marginBottom: 48 }}>
            <span className="pill">How it works</span>
            <h2 style={{ fontSize: 36, marginTop: 14 }}>From document to deployed bot in minutes</h2>
          </div>
          <div className="grid grid-3">
            <div className="card card-hover feature fade-up">
              <div className="icon-chip"><Icon path={ICONS.upload} /></div>
              <h3>1 · Ingest</h3>
              <p>Drop in a PDF or a URL. We extract, chunk, and embed it into a private vector store — one per project.</p>
            </div>
            <div className="card card-hover feature fade-up d1">
              <div className="icon-chip"><Icon path={ICONS.chat} /></div>
              <h3>2 · Chat</h3>
              <p>Ask in plain English. Answers are grounded strictly in your content and come back with the source passages.</p>
            </div>
            <div className="card card-hover feature fade-up d2">
              <div className="icon-chip"><Icon path={ICONS.embed} /></div>
              <h3>3 · Embed</h3>
              <p>Copy one <code>&lt;script&gt;</code> tag and the chat bubble appears on your own site, scoped to your project key.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="section" style={{ background: "var(--bg-2)" }}>
        <div className="container">
          <div className="grid grid-3">
            <div className="card feature"><div className="icon-chip"><Icon path={ICONS.shield} /></div>
              <h3>Grounded &amp; cited</h3><p>No hallucinated answers — every response is backed by retrieved passages from your sources.</p></div>
            <div className="card feature"><div className="icon-chip"><Icon path={ICONS.bolt} /></div>
              <h3>No vendor lock-in</h3><p>Local embeddings by default; swap the generation model (OpenAI, Groq, …) with a single env var.</p></div>
            <div className="card feature"><div className="icon-chip"><Icon path={ICONS.lock} /></div>
              <h3>Private by design</h3><p>Per-project isolation, domain allowlisting, and rate limits on the public widget API.</p></div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <div className="center" style={{ marginBottom: 40 }}>
            <span className="pill">Built for finance &amp; compliance</span>
            <h2 style={{ fontSize: 34, marginTop: 14 }}>Your higher-value flagship use case</h2>
            <p className="muted" style={{ maxWidth: 620, margin: "12px auto 0" }}>
              Make sense of complaints, policies, and filings — Product, Support, and
              Compliance get evidence-backed answers in seconds. The engine stays fully generic.
            </p>
          </div>
          <div className="grid grid-3">
            <div className="card card-hover"><strong>Complaint analysis</strong>
              <p className="muted">Spot emerging issues across thousands of narratives.</p><span className="pill">sample dataset</span></div>
            <div className="card card-hover"><strong>Policy &amp; handbook Q&amp;A</strong>
              <p className="muted">Answer staff questions from internal policy PDFs.</p><span className="pill">sample dataset</span></div>
            <div className="card card-hover"><strong>Product docs assistant</strong>
              <p className="muted">A support bot grounded in your documentation.</p><span className="pill">sample dataset</span></div>
          </div>
          <div className="center" style={{ marginTop: 44 }}>
            <Link className="btn btn-primary btn-lg" href="/signup">Start building free →</Link>
          </div>
        </div>
      </section>

      <footer className="footer">
        <div className="container row" style={{ justifyContent: "space-between" }}>
          <div className="brand" style={{ fontSize: 16 }}><Logo /> Grounded</div>
          <span>A deterministic, no-lock-in RAG platform.</span>
        </div>
      </footer>
    </main>
  );
}
