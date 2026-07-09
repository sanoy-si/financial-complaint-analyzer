"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "motion/react";
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
      <motion.div
        className="auth-card"
        initial={{ opacity: 0, y: 24, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      >
        <Link href="/" className="brand" style={{ justifyContent: "center", marginBottom: 8 }}>
          <Logo /> Grounded
        </Link>
        <h1 style={{ fontSize: 26, textAlign: "center", marginTop: 16 }}>Welcome back</h1>
        <p className="muted center" style={{ marginTop: 6 }}>Log in to your workspace</p>

        <form onSubmit={onSubmit} className="stack" style={{ marginTop: 28 }}>
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
          {error && (
            <motion.p
              className="error"
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
            >
              {error}
            </motion.p>
          )}
          <button className="btn btn-primary btn-block btn-lg" type="submit" disabled={loading}>
            {loading ? "Logging in…" : "Log in →"}
          </button>
        </form>

        <p className="muted center" style={{ marginTop: 20, fontSize: 14 }}>
          No account?{" "}
          <Link href="/signup" style={{ color: "#a78bfa", fontWeight: 600 }}>Sign up free</Link>
        </p>
      </motion.div>
    </main>
  );
}
