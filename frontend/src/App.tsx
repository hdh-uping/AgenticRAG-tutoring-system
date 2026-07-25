import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import {
  BookOpen,
  Check,
  ChevronDown,
  CircleUserRound,
  Database,
  FileText,
  GraduationCap,
  LoaderCircle,
  LogOut,
  Menu,
  MessageSquare,
  Network,
  Pencil,
  Plus,
  Search,
  Send,
  Settings2,
  ShieldCheck,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { ApiError, api, clearAuth, getStoredAuth, storeAuth } from "./api";
import { normalizeMathDelimiters } from "./markdown";
import type {
  ChunkDetail,
  Message,
  Preferences,
  Session,
  Source,
} from "./types";

const DEFAULT_PREFS: Preferences = {
  depth: "intermediate",
  show_code: "full",
  style: "casual",
  response_length: "balanced",
};

const SUGGESTIONS = [
  { icon: BookOpen, label: "顺序表是什么？", note: "检索教材定义与存储特点" },
  { icon: Network, label: "栈和队列有什么区别？", note: "结合图谱比较核心操作" },
  { icon: Database, label: "数组如何按下标访问元素？", note: "从公式与地址计算开始" },
];

function BrandLockup({ light = false }: { light?: boolean }) {
  return (
    <div className={`brand-lockup ${light ? "brand-lockup-light" : ""}`}>
      <div className="brand-mark" aria-hidden="true"><span>AR</span></div>
      <div>
        <strong>AgenticRAG</strong>
        <span>智慧教学系统</span>
      </div>
    </div>
  );
}

function formatTime(value?: string) {
  if (!value) return "";
  const normalized = value.includes("T") ? value : `${value.replace(" ", "T")}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return value;
  const now = new Date();
  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  }
  return date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "操作失败，请稍后重试";
}

type AuthScreenProps = {
  backendOnline: boolean | null;
  onAuthenticated: (userId: string) => void;
};

function AuthScreen({ backendOnline, onAuthenticated }: AuthScreenProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const auth = mode === "login"
        ? await api.login(userId.trim(), password)
        : await api.register(userId.trim(), password);
      storeAuth(auth);
      onAuthenticated(auth.user_id);
    } catch (requestError) {
      setError(errorMessage(requestError));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="auth-shell">
      <section className="auth-story" aria-label="产品介绍">
        <BrandLockup light />
        <div className="story-copy">
          <p className="eyebrow">AGENTIC RAG · EVIDENCE FIRST</p>
          <h1>让教材证据，<br />驱动每一次<br />教学决策。</h1>
          <p>教学 Agent 自主规划检索与回答，推荐 Agent 沿知识图谱延伸学习方向。</p>
        </div>
        <div className="story-features">
          <div><Sparkles size={18} /><span><b>双 Agent 协作</b>教学与推荐职责独立、流程可追踪</span></div>
          <div><Database size={18} /><span><b>Agentic 检索</b>按问题动态选择教材或知识图谱</span></div>
          <div><ShieldCheck size={18} /><span><b>证据约束</b>回答附带知识块、页码与执行轨迹</span></div>
        </div>
        <div className="flow-preview" aria-label="双 Agent 工作流">
          <div className="flow-preview-head">
            <span>LANGGRAPH ORCHESTRATION</span>
            <b><i /> RUNNING</b>
          </div>
          <div className="flow-preview-body">
            <div className="flow-node teaching-node">
              <small>01 · PLAN & RETRIEVE</small>
              <strong><GraduationCap size={17} /> 教学 Agent</strong>
              <span>混合检索 · 图谱查询 · 答案验证</span>
            </div>
            <div className="flow-arrow"><span /><span /><span /></div>
            <div className="flow-node recommendation-node">
              <small>02 · CONNECT & GUIDE</small>
              <strong><Network size={17} /> 推荐 Agent</strong>
              <span>关联候选 · 语境筛选 · 学习建议</span>
            </div>
          </div>
          <div className="flow-sources">
            <span><BookOpen size={14} /> 教材知识库</span>
            <span><Database size={14} /> Milvus</span>
            <span><Network size={14} /> Neo4j</span>
          </div>
        </div>
      </section>

      <section className="auth-panel">
        <div className="auth-card">
          <div className={`backend-chip ${backendOnline === false ? "offline" : ""}`}>
            <span />
            {backendOnline === null ? "正在检查服务" : backendOnline ? "教学服务已连接" : "教学服务未连接"}
          </div>
          <p className="eyebrow">AGENTICRAG WORKSPACE</p>
          <h2>{mode === "login" ? "进入教学工作台" : "创建学习账户"}</h2>
          <p>{mode === "login" ? "继续你的数据结构学习与 Agent 会话" : "保存学习会话、教材证据与回答偏好"}</p>

          <div className="auth-tabs" role="tablist" aria-label="认证方式">
            <button className={mode === "login" ? "active" : ""} onClick={() => { setMode("login"); setError(""); }} type="button">登录</button>
            <button className={mode === "register" ? "active" : ""} onClick={() => { setMode("register"); setError(""); }} type="button">注册</button>
          </div>

          <form onSubmit={submit} className="auth-form">
            <label>
              用户名
              <input
                autoComplete="username"
                value={userId}
                onChange={(event) => setUserId(event.target.value)}
                placeholder="3–64 位字母、数字或下划线"
                pattern="[A-Za-z0-9_.-]+"
                minLength={3}
                maxLength={64}
                required
              />
            </label>
            <label>
              密码
              <input
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="至少 8 个字符"
                minLength={8}
                maxLength={128}
                required
              />
            </label>
            {error && <div className="form-error" role="alert">{error}</div>}
            <button className="primary-button auth-submit" disabled={submitting || backendOnline === false}>
              {submitting ? <LoaderCircle className="spin" size={18} /> : <ShieldCheck size={18} />}
              {mode === "login" ? "登录并继续" : "创建账户"}
            </button>
          </form>
          <p className="privacy-note"><ShieldCheck size={14} /> 账户用于同步会话历史与学习偏好</p>
        </div>
      </section>
    </main>
  );
}

type SidebarProps = {
  open: boolean;
  userId: string;
  sessions: Session[];
  activeId: string;
  loading: boolean;
  onClose: () => void;
  onNew: () => void;
  onSelect: (session: Session) => void;
  onRename: (session: Session) => void;
  onDelete: (session: Session) => void;
  onPreferences: () => void;
  onLogout: () => void;
};

function Sidebar(props: SidebarProps) {
  const [query, setQuery] = useState("");
  const filtered = props.sessions.filter((session) =>
    session.title.toLowerCase().includes(query.toLowerCase()),
  );

  return (
    <>
      {props.open && <button className="sidebar-scrim" onClick={props.onClose} aria-label="关闭会话栏" />}
      <aside className={`sidebar ${props.open ? "open" : ""}`}>
        <div className="sidebar-top">
          <BrandLockup />
          <button className="mobile-close icon-button" onClick={props.onClose} aria-label="关闭"><X size={19} /></button>
        </div>

        <button className="new-chat-button" onClick={props.onNew}>
          <Plus size={18} /> 新建对话
        </button>

        <label className="search-box">
          <Search size={16} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索历史对话" aria-label="搜索历史对话" />
        </label>

        <div className="session-section-label"><span>最近对话</span><span>{props.sessions.length}</span></div>
        <nav className="session-list" aria-label="历史会话">
          {props.loading && <div className="session-placeholder"><LoaderCircle className="spin" size={18} /> 正在加载</div>}
          {!props.loading && filtered.length === 0 && (
            <div className="session-empty">{query ? "没有匹配的对话" : "还没有历史对话"}</div>
          )}
          {filtered.map((session) => (
            <div className={`session-row ${props.activeId === session.session_id ? "active" : ""}`} key={session.session_id}>
              <button className="session-main" onClick={() => props.onSelect(session)}>
                <MessageSquare size={16} />
                <span><b>{session.title}</b><small>{formatTime(session.updated_at)} · {session.message_count} 条消息</small></span>
              </button>
              <div className="session-actions">
                <button onClick={() => props.onRename(session)} aria-label={`重命名 ${session.title}`}><Pencil size={14} /></button>
                <button onClick={() => props.onDelete(session)} aria-label={`删除 ${session.title}`}><Trash2 size={14} /></button>
              </div>
            </div>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button onClick={props.onPreferences}><Settings2 size={17} /><span>回答偏好</span></button>
          <div className="account-row">
            <div className="account-avatar">{props.userId.slice(0, 1).toUpperCase()}</div>
              <div><strong>{props.userId}</strong><span>AgenticRAG 学习账户</span></div>
            <button className="logout-button" onClick={props.onLogout} aria-label="退出登录"><LogOut size={17} /></button>
          </div>
        </div>
      </aside>
    </>
  );
}

type MessageBubbleProps = {
  message: Message;
  onSource: (source: Source) => void;
};

function MessageBubble({ message, onSource }: MessageBubbleProps) {
  const sources = message.metadata?.sources || [];
  const trace = message.trace || [
    ...(message.metadata?.trace || []),
    ...(message.metadata?.recommendation_trace || []),
  ];
  return (
    <article className={`message ${message.role}`}>
      <div className="message-avatar">
        {message.role === "assistant" ? <GraduationCap size={19} /> : <CircleUserRound size={19} />}
      </div>
      <div className="message-body">
        <div className="message-meta">
          <strong>{message.role === "assistant" ? "AgenticRAG · 教学 Agent" : "你"}</strong>
          {message.created_at && <time>{formatTime(message.created_at)}</time>}
        </div>
        <div className="message-content">
          {message.role === "assistant" ? (
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={{ a: (props) => <a {...props} target="_blank" rel="noreferrer" /> }}
            >
              {normalizeMathDelimiters(message.content)}
            </ReactMarkdown>
          ) : <p>{message.content}</p>}
        </div>
        {message.role === "assistant" && (sources.length > 0 || trace.length > 0) && (
          <div className="evidence-row">
            {sources.map((source) => (
              <button key={`${source.kind}-${source.id}`} onClick={() => onSource(source)}>
                {source.kind === "chunk" ? <FileText size={14} /> : <Network size={14} />}
                {source.kind === "chunk" && source.page_num
                  ? `${source.document || source.id.split("_chunk_")[0]} · 第 ${source.page_num} 页`
                  : source.id}
              </button>
            ))}
            {trace.length > 0 && (
              <details className="trace-details">
                <summary><Sparkles size={14} /> 查看双 Agent 路径 <ChevronDown size={13} /></summary>
                <div className="trace-list">
                  {trace.map((item, index) => (
                    <div key={`${item.turn}-${index}`}>
                      <span>{item.turn}</span>
                      <div className="trace-copy">
                        <p><b>{item.agent === "recommendation" ? "推荐 Agent" : "教学 Agent"} · {item.action}</b>{item.reason_summary && ` · ${item.reason_summary}`}</p>
                        {item.plan?.requirements?.length
                          ? <small>完成条件：{item.plan.requirements.join("；")}</small>
                          : null}
                        {item.skill_instructions_loaded
                          ? <small>Skill 详情：首次调用时按需加载</small>
                          : null}
                        {item.status
                          ? <small>验证结果：{item.status}{item.issues?.length ? ` · ${item.issues.join("；")}` : ""}</small>
                          : null}
                        {item.remaining_issues?.length
                          ? <small>仍待处理：{item.remaining_issues.join("；")}</small>
                          : null}
                      </div>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}
      </div>
    </article>
  );
}

type PreferencesModalProps = {
  value: Preferences;
  saving: boolean;
  onClose: () => void;
  onSave: (value: Preferences) => void;
};

function PreferencesModal({ value, saving, onClose, onSave }: PreferencesModalProps) {
  const [draft, setDraft] = useState(value);
  function update<K extends keyof Preferences>(key: K, next: Preferences[K]) {
    setDraft((current) => ({ ...current, [key]: next }));
  }
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="modal-card preferences-modal" role="dialog" aria-modal="true" aria-labelledby="prefs-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <div><p className="eyebrow">LEARNING PREFERENCES</p><h2 id="prefs-title">回答偏好</h2></div>
          <button className="icon-button" onClick={onClose} aria-label="关闭"><X size={19} /></button>
        </div>
        <p className="modal-intro">这些设置会跨会话保存，并注入每次教学回答。你也可以在对话中自然表达偏好。</p>
        <div className="preference-grid">
          <label>讲解深度<select value={draft.depth} onChange={(event) => update("depth", event.target.value as Preferences["depth"])}><option value="beginner">初学者 · 多铺垫与类比</option><option value="intermediate">标准 · 正常节奏</option><option value="advanced">进阶 · 直击核心</option></select></label>
          <label>代码展示<select value={draft.show_code} onChange={(event) => update("show_code", event.target.value as Preferences["show_code"])}><option value="full">完整代码</option><option value="idea">只讲思路</option></select></label>
          <label>表达风格<select value={draft.style} onChange={(event) => update("style", event.target.value as Preferences["style"])}><option value="casual">亲切口语</option><option value="academic">正式学术</option></select></label>
          <label>回答长度<select value={draft.response_length} onChange={(event) => update("response_length", event.target.value as Preferences["response_length"])}><option value="concise">简洁</option><option value="balanced">适中</option><option value="detailed">详细</option></select></label>
        </div>
        <div className="modal-actions"><button className="secondary-button" onClick={onClose}>取消</button><button className="primary-button" disabled={saving} onClick={() => onSave(draft)}>{saving ? <LoaderCircle className="spin" size={17} /> : <Check size={17} />} 保存偏好</button></div>
      </section>
    </div>
  );
}

type SourceModalProps = {
  source: Source;
  detail: ChunkDetail | null;
  loading: boolean;
  onClose: () => void;
};

function SourceModal({ source, detail, loading, onClose }: SourceModalProps) {
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="modal-card source-modal" role="dialog" aria-modal="true" aria-labelledby="source-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <div><p className="eyebrow">EVIDENCE SOURCE</p><h2 id="source-title">{source.kind === "chunk" ? "教材原文" : "知识图谱节点"}</h2></div>
          <button className="icon-button" onClick={onClose} aria-label="关闭"><X size={19} /></button>
        </div>
        <div className="source-meta">
          <span>{source.kind === "chunk" ? <FileText size={15} /> : <Network size={15} />}{source.id}</span>
          {source.page_num && <span>第 {source.page_num} 页</span>}
          {source.rerank_score !== undefined && <span>相关度 {(source.rerank_score * 100).toFixed(1)}%</span>}
          {source.node_type && <span>{source.node_type}</span>}
        </div>
        {loading ? <div className="source-loading"><LoaderCircle className="spin" /> 正在读取教材原文</div> : detail ? (
          <div className="source-content"><small>{detail.document}{detail.header_path ? ` · ${detail.header_path}` : ""}</small><p>{detail.text}</p></div>
        ) : (
          <div className="source-content graph-source"><Network size={28} /><p>该回答使用了知识图谱中的“{source.id}”节点。图谱节点包含结构化描述、操作步骤或复杂度关系。</p></div>
        )}
      </section>
    </div>
  );
}

export default function App() {
  const stored = useMemo(() => getStoredAuth(), []);
  const [authenticated, setAuthenticated] = useState(Boolean(stored.token));
  const [userId, setUserId] = useState(stored.userId);
  const [checkingAuth, setCheckingAuth] = useState(Boolean(stored.token));
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [question, setQuestion] = useState("");
  const [sending, setSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [preferences, setPreferences] = useState(DEFAULT_PREFS);
  const [preferencesOpen, setPreferencesOpen] = useState(false);
  const [preferencesSaving, setPreferencesSaving] = useState(false);
  const [sourceState, setSourceState] = useState<{ source: Source; detail: ChunkDetail | null; loading: boolean } | null>(null);
  const [toast, setToast] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    api.health().then(() => setBackendOnline(true)).catch(() => setBackendOnline(false));
    const expire = () => {
      setAuthenticated(false);
      setUserId("");
      setSessions([]);
      setMessages([]);
      setToast("登录状态已失效，请重新登录");
    };
    window.addEventListener("auth-expired", expire);
    return () => window.removeEventListener("auth-expired", expire);
  }, []);

  useEffect(() => {
    if (!authenticated) {
      setCheckingAuth(false);
      return;
    }
    let cancelled = false;
    async function bootstrap() {
      setSessionsLoading(true);
      try {
        const [me, sessionResult, prefResult] = await Promise.all([
          api.me(), api.listSessions(), api.getPreferences(),
        ]);
        if (cancelled) return;
        setUserId(me.user_id);
        setSessions(sessionResult.sessions);
        setPreferences(prefResult.prefs);
      } catch (error) {
        if (!(error instanceof ApiError && error.status === 401)) setToast(errorMessage(error));
      } finally {
        if (!cancelled) {
          setSessionsLoading(false);
          setCheckingAuth(false);
        }
      }
    }
    bootstrap();
    return () => { cancelled = true; };
  }, [authenticated]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 3600);
    return () => window.clearTimeout(timer);
  }, [toast]);

  async function refreshSessions(preferredId?: string) {
    const result = await api.listSessions();
    setSessions(result.sessions);
    if (preferredId) {
      const next = result.sessions.find((item) => item.session_id === preferredId);
      if (next) setActiveSession(next);
    }
  }

  async function selectSession(session: Session) {
    setSidebarOpen(false);
    setActiveSession(session);
    setHistoryLoading(true);
    try {
      const detail = await api.getSession(session.session_id);
      setMessages(detail.messages);
    } catch (error) {
      setToast(errorMessage(error));
    } finally {
      setHistoryLoading(false);
    }
  }

  function newConversation() {
    setActiveSession(null);
    setMessages([]);
    setQuestion("");
    setSidebarOpen(false);
    window.setTimeout(() => inputRef.current?.focus(), 0);
  }

  async function renameSession(session: Session) {
    const title = window.prompt("输入新的会话标题", session.title)?.trim();
    if (!title || title === session.title) return;
    try {
      const renamed = await api.renameSession(session.session_id, title);
      setSessions((current) => current.map((item) => item.session_id === renamed.session_id ? renamed : item));
      if (activeSession?.session_id === renamed.session_id) setActiveSession(renamed);
    } catch (error) { setToast(errorMessage(error)); }
  }

  async function removeSession(session: Session) {
    if (!window.confirm(`确定删除“${session.title}”及其全部消息吗？`)) return;
    try {
      await api.deleteSession(session.session_id);
      setSessions((current) => current.filter((item) => item.session_id !== session.session_id));
      if (activeSession?.session_id === session.session_id) newConversation();
      setToast("会话已删除");
    } catch (error) { setToast(errorMessage(error)); }
  }

  async function sendQuestion(value = question) {
    const clean = value.trim();
    if (!clean || sending) return;
    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content: clean };
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setSending(true);
    try {
      const result = await api.chat(clean, activeSession?.session_id || "");
      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: result.answer,
        trace: [
          ...result.trace.map((item) => ({ ...item, agent: "teaching" as const })),
          ...result.recommendation_trace,
        ],
        metadata: {
          sources: result.sources,
          concepts: result.concepts_involved,
          iterations: result.iterations,
        },
      };
      setMessages((current) => [...current, assistantMessage]);
      await Promise.all([
        refreshSessions(result.session_id),
        api.getPreferences().then((response) => setPreferences(response.prefs)),
      ]);
    } catch (error) {
      setToast(errorMessage(error));
    } finally {
      setSending(false);
      window.setTimeout(() => inputRef.current?.focus(), 0);
    }
  }

  function inputKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendQuestion();
    }
  }

  async function savePreferences(next: Preferences) {
    setPreferencesSaving(true);
    try {
      const response = await api.updatePreferences(next);
      setPreferences(response.prefs);
      setPreferencesOpen(false);
      setToast("回答偏好已保存");
    } catch (error) { setToast(errorMessage(error)); }
    finally { setPreferencesSaving(false); }
  }

  async function openSource(source: Source) {
    setSourceState({ source, detail: null, loading: source.kind === "chunk" });
    if (source.kind !== "chunk") return;
    try {
      const detail = await api.getSource(source.id);
      setSourceState({ source, detail, loading: false });
    } catch (error) {
      setSourceState({ source, detail: null, loading: false });
      setToast(errorMessage(error));
    }
  }

  async function signOut() {
    try { await api.logout(); } catch { /* 本地仍然清理令牌。 */ }
    clearAuth();
    setAuthenticated(false);
    setUserId("");
    setSessions([]);
    setMessages([]);
    setActiveSession(null);
  }

  if (checkingAuth) {
    return <div className="app-loading"><div className="brand-mark"><span>AR</span></div><LoaderCircle className="spin" /><span>正在恢复 AgenticRAG智慧教学系统</span></div>;
  }
  if (!authenticated) {
    return <AuthScreen backendOnline={backendOnline} onAuthenticated={(nextUser) => { setUserId(nextUser); setAuthenticated(true); setCheckingAuth(true); }} />;
  }

  return (
    <div className="app-shell">
      <Sidebar
        open={sidebarOpen}
        userId={userId}
        sessions={sessions}
        activeId={activeSession?.session_id || ""}
        loading={sessionsLoading}
        onClose={() => setSidebarOpen(false)}
        onNew={newConversation}
        onSelect={selectSession}
        onRename={renameSession}
        onDelete={removeSession}
        onPreferences={() => setPreferencesOpen(true)}
        onLogout={signOut}
      />

      <main className="chat-workspace">
        <header className="chat-header">
          <button className="menu-button icon-button" onClick={() => setSidebarOpen(true)} aria-label="打开会话栏"><Menu size={20} /></button>
          <div><span>{activeSession ? "ACTIVE LEARNING SESSION" : "NEW AGENT SESSION"}</span><h1>{activeSession?.title || "AgenticRAG智慧教学系统"}</h1></div>
          <div className="header-status">
            <span />
            <div><b>双 Agent 在线</b><small>教材库 · 向量库 · 知识图谱</small></div>
          </div>
        </header>

        <section className="chat-scroll" aria-live="polite">
          <div className="chat-column">
            {historyLoading ? (
              <div className="center-state"><LoaderCircle className="spin" /><p>正在加载历史对话</p></div>
            ) : messages.length === 0 ? (
              <div className="welcome-state">
                <div className="welcome-badge"><span /> AGENTICRAG LEARNING WORKSPACE</div>
                <h2>让 Agent 先取证，<br />再回答。</h2>
                <p>教学 Agent 会自主选择教材检索或图谱查询，完成回答后，推荐 Agent 将继续关联下一步学习内容。</p>
                <div className="agent-overview">
                  <div>
                    <span className="agent-index">01</span>
                    <GraduationCap size={18} />
                    <p><b>教学 Agent</b><small>规划 · 检索 · 验证 · 回答</small></p>
                    <i>ACTIVE</i>
                  </div>
                  <span className="agent-link" />
                  <div>
                    <span className="agent-index">02</span>
                    <Network size={18} />
                    <p><b>推荐 Agent</b><small>关联 · 筛选 · 学习建议</small></p>
                    <i>READY</i>
                  </div>
                </div>
                <div className="suggestion-grid">
                  {SUGGESTIONS.map(({ icon: Icon, label, note }) => (
                    <button key={label} onClick={() => void sendQuestion(label)}>
                      <Icon size={20} /><span><b>{label}</b><small>{note}</small></span><Send size={15} />
                    </button>
                  ))}
                </div>
                <div className="trust-line"><ShieldCheck size={15} /> 知识性回答必须取得教材或图谱证据后才能生成</div>
              </div>
            ) : (
              <div className="message-list">
                {messages.map((message) => <MessageBubble key={message.id} message={message} onSource={openSource} />)}
                {sending && (
                  <article className="message assistant typing-message">
                    <div className="message-avatar"><GraduationCap size={19} /></div>
                    <div className="message-body"><div className="message-meta"><strong>AgenticRAG · 教学 Agent</strong></div><div className="thinking"><span /><span /><span /></div><p>正在规划检索路径并核对教材证据…</p></div>
                  </article>
                )}
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </section>

        <footer className="composer-area">
          <div className="composer">
            <textarea
              ref={inputRef}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={inputKeyDown}
              placeholder="向教学 Agent 提出一个数据结构问题…"
              maxLength={4000}
              rows={1}
              disabled={sending}
              aria-label="输入问题"
            />
            <div className="composer-bottom">
              <span><Sparkles size={14} /> AgenticRAG · Enter 发送</span>
              <button onClick={() => void sendQuestion()} disabled={!question.trim() || sending} aria-label="发送问题">
                {sending ? <LoaderCircle className="spin" size={18} /> : <Send size={18} />}
              </button>
            </div>
          </div>
          <p>AgenticRAG 会展示教材来源与执行路径，重要结论仍建议结合原文核对。</p>
        </footer>
      </main>

      {preferencesOpen && <PreferencesModal value={preferences} saving={preferencesSaving} onClose={() => setPreferencesOpen(false)} onSave={savePreferences} />}
      {sourceState && <SourceModal {...sourceState} onClose={() => setSourceState(null)} />}
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}
