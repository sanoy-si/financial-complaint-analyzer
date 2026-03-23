"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
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
    <main className="container" style={{ maxWidth: 400, paddingTop: 64 }}>
      <h1>Create your account</h1>
      <form onSubmit={onSubmit} className="card">
        <label>Email</label>
        <input className="input" type="email" value={email}
          onChange={(e) => setEmail(e.target.value)} required />
        <label style={{ marginTop: 12, display: "block" }}>Password</label>
        <input className="input" type="password" value={password}
          onChange={(e) => setPassword(e.target.value)} required />
        {error && <p className="error">{error}</p>}
        <button className="btn" type="submit" disabled={loading} style={{ marginTop: 16 }}>
          {loading ? "Creating…" : "Sign up"}
        </button>
      </form>
      <p className="muted" style={{ marginTop: 12 }}>
        Already have an account? <Link href="/login">Log in</Link>
      </p>
    </main>
  );
}
