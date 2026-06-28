"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Logo } from "@/components/Logo";
import { api, clearToken, getToken, type Project } from "@/lib/api";

export default function DashboardPage() {
  const router = useRouter();
  const [projects, setProjects] = useState<Project[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!getToken()) {
      router.push("/login");
      return;
    }
    api.listProjects().then(setProjects).catch((e) => setError(e.message)).finally(() => setLoading(false));
  }, [router]);

  async function create(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    try {
      const project = await api.createProject(name.trim());
      setProjects((p) => [...p, project]);
      setName("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create project");
    }
  }

  function logout() {
    clearToken();
    router.push("/login");
  }

  return (
    <div className="app-bg">
      <nav className="nav">
        <div className="nav-inner">
          <Link href="/dashboard" className="brand"><Logo /> Grounded</Link>
          <button className="btn btn-ghost" onClick={logout}>Log out</button>
        </div>
      </nav>

      <div className="container" style={{ paddingTop: 40, paddingBottom: 60 }}>
        <h1 style={{ fontSize: 32 }}>Your chatbots</h1>
        <p className="muted" style={{ marginTop: 4 }}>Each project is an isolated knowledge base with its own embed key.</p>

        <form onSubmit={create} className="row fade-up" style={{ margin: "24px 0 32px", maxWidth: 480 }}>
          <input className="input" placeholder="e.g. Support docs bot" value={name}
            onChange={(e) => setName(e.target.value)} />
          <button className="btn btn-primary" type="submit">Create</button>
        </form>

        {error && <p className="error">{error}</p>}

        {loading ? (
          <p className="muted">Loading…</p>
        ) : projects.length === 0 ? (
          <div className="empty fade-up">
            <div style={{ fontSize: 40 }}>🪄</div>
            <p style={{ marginTop: 8 }}>No chatbots yet. Create your first one above.</p>
          </div>
        ) : (
          <div className="grid grid-3">
            {projects.map((p, i) => (
              <Link key={p.id} href={`/projects/${p.id}`}
                className={`card card-hover proj-card fade-up d${(i % 3) + 1}`}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <div className="avatar">{p.name.charAt(0).toUpperCase()}</div>
                  <span className="pill">active</span>
                </div>
                <h3 style={{ fontSize: 18, marginTop: 14 }}>{p.name}</h3>
                <div className="key">{p.public_key}</div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
