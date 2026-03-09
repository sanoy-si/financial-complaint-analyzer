/* Thin typed client for the Grounded backend. */

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const V1 = `${API_BASE}/api/v1`;

const TOKEN_KEY = "grounded_token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string) {
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  window.localStorage.removeItem(TOKEN_KEY);
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${V1}${path}`, { ...options, headers });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export interface Project {
  id: string;
  name: string;
  public_key: string;
  settings: Record<string, unknown>;
  created_at: string;
}

export interface DocumentItem {
  id: string;
  source_type: string;
  source_ref: string;
  status: string;
  error?: string | null;
  created_at: string;
}

export interface Source {
  content: string;
  score: number;
  document_id?: string | null;
}

export interface ChatResult {
  answer: string;
  sources: Source[];
  session_id: string;
}

export const api = {
  signup: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/signup", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  login: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  me: () => request<{ id: string; email: string }>("/auth/me"),

  listProjects: () => request<Project[]>("/projects"),
  createProject: (name: string) =>
    request<Project>("/projects", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  getProject: (id: string) => request<Project>(`/projects/${id}`),

  listDocuments: (projectId: string) =>
    request<DocumentItem[]>(`/projects/${projectId}/documents`),
  ingestUrl: (projectId: string, url: string) =>
    request<DocumentItem>(`/projects/${projectId}/documents/url`, {
      method: "POST",
      body: JSON.stringify({ url }),
    }),
  uploadPdf: async (projectId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    const headers: Record<string, string> = {};
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res = await fetch(`${V1}/projects/${projectId}/documents/pdf`, {
      method: "POST",
      headers,
      body: form,
    });
    if (!res.ok) throw new Error(`Upload failed (${res.status})`);
    return res.json() as Promise<DocumentItem>;
  },

  chat: (projectId: string, question: string, sessionId?: string) =>
    request<ChatResult>(`/projects/${projectId}/chat`, {
      method: "POST",
      body: JSON.stringify({ question, session_id: sessionId }),
    }),
};

export { API_BASE };
