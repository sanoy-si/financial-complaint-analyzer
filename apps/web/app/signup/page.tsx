"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Logo } from "@/components/Logo";
import { api, setToken } from "@/lib/api";

export default function SignupPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }
    setLoading(true);
    try {
      const { access_token } = await api.signup(email, password);
      setToken(access_token);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Signup failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-wrap">
      <div className="auth-card fade-up">
        <Link href="/" className="brand" style={{ justifyContent: "center", marginBottom: 8 }}>
          <Logo /> Grounded
        </Link>
        <h1 style={{ fontSize: 26, textAlign: "center" }}>Create your account</h1>
        <p className="muted center" style={{ marginTop: 4 }}>Spin up your first chatbot in minutes</p>
        <form onSubmit={onSubmit} className="stack" style={{ marginTop: 24 }}>
          <div className="field">
            <label className="label">Email</label>
            <input className="input" type="email" value={email}
              onChange={(e) => setEmail(e.target.value)} required placeholder="you@company.com" />
          </div>
          <div className="field">
            <label className="label">Password</label>
            <input className="input" type="password" value={password}
              onChange={(e) => setPassword(e.target.value)} required placeholder="At least 8 characters" />
          </div>
          {error && <p className="error">{error}</p>}
          <button className="btn btn-primary btn-block btn-lg" type="submit" disabled={loading}>
            {loading ? "Creating…" : "Create account"}
          </button>
        </form>
        <p className="muted center" style={{ marginTop: 18 }}>
          Already have an account? <Link href="/login" style={{ color: "var(--violet)", fontWeight: 600 }}>Log in</Link>
        </p>
      </div>
    </main>
  );
}
