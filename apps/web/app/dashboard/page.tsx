"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
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
    api
      .listProjects()
      .then(setProjects)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
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
    <main>
      <nav className="nav">
        <Link href="/dashboard"><strong>Grounded</strong></Link>
        <button className="btn secondary" onClick={logout}>Log out</button>
      </nav>
      <div className="container">
        <h1>Your chatbots</h1>
        <form onSubmit={create} className="row" style={{ marginBottom: 24 }}>
          <input className="input" placeholder="New project name" value={name}
            onChange={(e) => setName(e.target.value)} />
          <button className="btn" type="submit">Create</button>
        </form>
        {error && <p className="error">{error}</p>}
        {loading ? (
          <p className="muted">Loading…</p>
        ) : projects.length === 0 ? (
          <p className="muted">No projects yet. Create one to get started.</p>
        ) : (
          <div className="grid">
            {projects.map((p) => (
              <Link key={p.id} href={`/projects/${p.id}`} className="card">
                <strong>{p.name}</strong>
                <p className="muted" style={{ fontSize: 12, wordBreak: "break-all" }}>
                  {p.public_key}
                </p>
              </Link>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
