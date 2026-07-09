"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion, AnimatePresence } from "motion/react";
import { Logo } from "@/components/Logo";
import { SpringCard, FadeIn, Stagger, StaggerItem } from "@/components/MotionWrap";
import { api, clearToken, getToken, type Project } from "@/lib/api";

export default function DashboardPage() {
  const router = useRouter();
  const [projects, setProjects] = useState<Project[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!getToken()) { router.push("/login"); return; }
    api.listProjects()
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

  function logout() { clearToken(); router.push("/login"); }

  return (
    <div className="app-bg">
      <nav className="nav">
        <div className="nav-inner">
          <Link href="/dashboard" className="brand"><Logo /> Grounded</Link>
          <button className="btn btn-ghost" onClick={logout}>Log out</button>
        </div>
      </nav>

      <div className="container" style={{ paddingTop: 48, paddingBottom: 72 }}>

        <FadeIn>
          <h1 style={{ fontSize: 34 }}>Your chatbots</h1>
          <p className="muted" style={{ marginTop: 6 }}>
            Each project is an isolated knowledge base with its own embed key.
          </p>
        </FadeIn>

        {/* create form */}
        <FadeIn delay={0.06}>
          <form onSubmit={create} className="row" style={{ margin: "28px 0 40px", maxWidth: 520 }}>
            <input
              className="input"
              placeholder="e.g. Support docs bot"
              value={name}
              onChange={(e) => setName(e.target.value)}
              style={{ flex: 1 }}
            />
            <button className="btn btn-primary" type="submit">
              + Create
            </button>
          </form>
          {error && <p className="error">{error}</p>}
        </FadeIn>

        {/* project grid */}
        {loading ? (
          <FadeIn>
            <p className="muted">Loading…</p>
          </FadeIn>
        ) : projects.length === 0 ? (
          <FadeIn>
            <div className="empty">
              <div style={{ fontSize: 44, marginBottom: 12 }}>🪄</div>
              <p>No chatbots yet. Create your first one above.</p>
            </div>
          </FadeIn>
        ) : (
          <Stagger className="grid grid-3">
            <AnimatePresence>
              {projects.map((p) => (
                <StaggerItem key={p.id}>
                  <SpringCard>
                    <Link href={`/projects/${p.id}`} className="card card-hover proj-card"
                      style={{ display: "block", height: "100%" }}>
                      <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
                        <div className="avatar">{p.name.charAt(0).toUpperCase()}</div>
                        <span className="pill">active</span>
                      </div>
                      <h3 style={{ fontSize: 18, marginTop: 16 }}>{p.name}</h3>
                      <div className="key">{p.public_key}</div>
                    </Link>
                  </SpringCard>
                </StaggerItem>
              ))}
            </AnimatePresence>
          </Stagger>
        )}
      </div>
    </div>
  );
}
