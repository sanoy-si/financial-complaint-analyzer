import Link from "next/link";

export default function LandingPage() {
  return (
    <main>
      <nav className="nav">
        <strong>Grounded</strong>
        <div className="row">
          <Link href="/login">Log in</Link>
          <Link className="btn" href="/signup">
            Get started
          </Link>
        </div>
      </nav>

      <section className="container" style={{ paddingTop: 64, paddingBottom: 48 }}>
        <h1 style={{ fontSize: 44, lineHeight: 1.1, maxWidth: 720 }}>
          Turn your documents into a chatbot your customers can actually trust.
        </h1>
        <p className="muted" style={{ fontSize: 18, maxWidth: 640 }}>
          Upload a PDF or paste a URL. Grounded ingests the content and answers
          questions using only your sources — then gives you an embeddable widget
          to drop on your own site with one line of code.
        </p>
        <div className="row" style={{ marginTop: 24 }}>
          <Link className="btn" href="/signup">
            Create your first bot
          </Link>
          <Link className="btn secondary" href="/login">
            I already have an account
          </Link>
        </div>
        <p className="muted" style={{ marginTop: 12, fontSize: 13 }}>
          Answers are grounded in your sources. Your documents are embedded
          locally and never leave your server.
        </p>
      </section>

      <section className="container">
        <h2>Built for finance &amp; compliance teams</h2>
        <p className="muted" style={{ maxWidth: 680 }}>
          The flagship use case: make sense of customer complaints, policies, and
          regulatory filings. Product, Support, and Compliance ask plain-English
          questions and get evidence-backed answers in seconds — but the engine is
          fully generic, so it works on any document set.
        </p>
        <div className="grid" style={{ marginTop: 16 }}>
          <div className="card">
            <strong>Complaint analysis</strong>
            <p className="muted">Spot emerging issues across thousands of narratives.</p>
            <span className="tag">sample dataset</span>
          </div>
          <div className="card">
            <strong>Policy &amp; handbook Q&amp;A</strong>
            <p className="muted">Answer staff questions from internal policy PDFs.</p>
            <span className="tag">sample dataset</span>
          </div>
          <div className="card">
            <strong>Product docs assistant</strong>
            <p className="muted">A support bot grounded in your documentation site.</p>
            <span className="tag">sample dataset</span>
          </div>
        </div>
      </section>

      <footer className="container muted" style={{ marginTop: 48, fontSize: 13 }}>
        Grounded · a deterministic, no-lock-in RAG platform.
      </footer>
    </main>
  );
}
