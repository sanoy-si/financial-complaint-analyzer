"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "motion/react";
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
    if (password.length < 8) { setError("Password must be at least 8 characters."); return; }
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
      <motion.div
        className="auth-card"
        initial={{ opacity: 0, y: 24, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      >
        <Link href="/" className="brand" style={{ justifyContent: "center", marginBottom: 8 }}>
          <Logo /> Grounded
        </Link>
        <h1 style={{ fontSize: 26, textAlign: "center", marginTop: 16 }}>Create your account</h1>
        <p className="muted center" style={{ marginTop: 6 }}>Spin up your first chatbot in minutes</p>

        <form onSubmit={onSubmit} className="stack" style={{ marginTop: 28 }}>
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
            {loading ? "Creating account…" : "Create account →"}
          </button>
        </form>

        <p className="muted center" style={{ marginTop: 20, fontSize: 14 }}>
          Already have an account?{" "}
          <Link href="/login" style={{ color: "#a78bfa", fontWeight: 600 }}>Log in</Link>
        </p>
      </motion.div>
    </main>
  );
}
