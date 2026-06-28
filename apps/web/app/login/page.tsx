"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Logo } from "@/components/Logo";
import { api, setToken } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const { access_token } = await api.login(email, password);
      setToken(access_token);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
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
        <h1 style={{ fontSize: 26, textAlign: "center" }}>Welcome back</h1>
        <p className="muted center" style={{ marginTop: 4 }}>Log in to your workspace</p>
        <form onSubmit={onSubmit} className="stack" style={{ marginTop: 24 }}>
          <div className="field">
            <label className="label">Email</label>
            <input className="input" type="email" value={email}
              onChange={(e) => setEmail(e.target.value)} required placeholder="you@company.com" />
          </div>
          <div className="field">
            <label className="label">Password</label>
            <input className="input" type="password" value={password}
              onChange={(e) => setPassword(e.target.value)} required placeholder="••••••••" />
          </div>
          {error && <p className="error">{error}</p>}
          <button className="btn btn-primary btn-block btn-lg" type="submit" disabled={loading}>
            {loading ? "Logging in…" : "Log in"}
          </button>
        </form>
        <p className="muted center" style={{ marginTop: 18 }}>
          No account? <Link href="/signup" style={{ color: "var(--violet)", fontWeight: 600 }}>Sign up</Link>
        </p>
      </div>
    </main>
  );
}
