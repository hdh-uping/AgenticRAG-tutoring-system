import type {
  AuthResponse,
  ChatResponse,
  ChunkDetail,
  Preferences,
  Session,
  SessionDetail,
} from "./types";

const TOKEN_KEY = "agenticrag_access_token";
const USER_KEY = "agenticrag_user_id";
const LEGACY_TOKEN_KEY = "zhixu_access_token";
const LEGACY_USER_KEY = "zhixu_user_id";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function getStoredAuth() {
  const token = localStorage.getItem(TOKEN_KEY) || localStorage.getItem(LEGACY_TOKEN_KEY) || "";
  const userId = localStorage.getItem(USER_KEY) || localStorage.getItem(LEGACY_USER_KEY) || "";
  if (token && !localStorage.getItem(TOKEN_KEY)) localStorage.setItem(TOKEN_KEY, token);
  if (userId && !localStorage.getItem(USER_KEY)) localStorage.setItem(USER_KEY, userId);
  return {
    token,
    userId,
  };
}

export function storeAuth(auth: AuthResponse) {
  localStorage.setItem(TOKEN_KEY, auth.access_token);
  localStorage.setItem(USER_KEY, auth.user_id);
}

export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(LEGACY_TOKEN_KEY);
  localStorage.removeItem(LEGACY_USER_KEY);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = localStorage.getItem(TOKEN_KEY);
  const headers = new Headers(init.headers);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (token) headers.set("Authorization", `Bearer ${token}`);

  let response: Response;
  try {
    response = await fetch(`/api${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, "无法连接后端，请确认 FastAPI 已启动");
  }

  if (response.status === 401 && token) {
    clearAuth();
    window.dispatchEvent(new Event("auth-expired"));
  }
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // 保留通用错误信息。
    }
    throw new ApiError(response.status, message);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/health"),
  register: (userId: string, password: string) =>
    request<AuthResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, password }),
    }),
  login: (userId: string, password: string) =>
    request<AuthResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, password }),
    }),
  me: () => request<{ user_id: string }>("/auth/me"),
  logout: () => request<void>("/auth/logout", { method: "POST" }),
  listSessions: () =>
    request<{ sessions: Session[] }>("/sessions?limit=100"),
  getSession: (sessionId: string) =>
    request<SessionDetail>(`/sessions/${encodeURIComponent(sessionId)}?limit=500`),
  createSession: (title = "新会话") =>
    request<Session>("/sessions", {
      method: "POST",
      body: JSON.stringify({ title }),
    }),
  renameSession: (sessionId: string, title: string) =>
    request<Session>(`/sessions/${encodeURIComponent(sessionId)}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),
  deleteSession: (sessionId: string) =>
    request<void>(`/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    }),
  chat: (question: string, sessionId = "") =>
    request<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify({ question, session_id: sessionId }),
    }),
  getPreferences: () =>
    request<{ prefs: Preferences }>("/prefs"),
  updatePreferences: (prefs: Preferences) =>
    request<{ prefs: Preferences }>("/prefs", {
      method: "PUT",
      body: JSON.stringify(prefs),
    }),
  getSource: (chunkId: string) =>
    request<ChunkDetail>(`/sources/${encodeURIComponent(chunkId)}`),
};
