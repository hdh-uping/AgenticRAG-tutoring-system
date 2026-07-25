export type Preferences = {
  depth: "beginner" | "intermediate" | "advanced";
  show_code: "full" | "idea";
  style: "casual" | "academic";
  response_length: "concise" | "balanced" | "detailed";
};

export type Source = {
  kind: "chunk" | "graph";
  id: string;
  page_num?: number | string;
  rerank_score?: number;
  node_type?: string;
  document?: string;
  skill: string;
};

export type TraceItem = {
  agent?: "teaching" | "recommendation";
  turn: number;
  action: string;
  reason_summary?: string;
  input?: string;
  result?: string;
  evidence_ids?: string[];
  rerank_scores?: number[];
  observation_preview?: string;
  status?: "pass" | "revise" | "retrieve_more";
  issues?: string[];
  missing_requirements?: string[];
  remaining_issues?: string[];
  skill_instructions_loaded?: boolean;
  plan?: {
    question_type: string;
    requirements: string[];
    steps?: { goal: string }[];
    checks?: string[];
  };
};

export type MessageMetadata = {
  sources?: Source[];
  concepts?: string[];
  inferred_prefs?: Partial<Preferences>;
  iterations?: number;
  trace?: TraceItem[];
  recommendation_trace?: TraceItem[];
};

export type Message = {
  id: number | string;
  role: "user" | "assistant";
  content: string;
  metadata?: MessageMetadata;
  created_at?: string;
  trace?: TraceItem[];
};

export type Session = {
  session_id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_message: string;
};

export type SessionDetail = Session & {
  messages: Message[];
  has_more: boolean;
};

export type ChatResponse = {
  answer: string;
  recommendation: string;
  session_id: string;
  trace: TraceItem[];
  recommendation_trace: TraceItem[];
  iterations: number;
  concepts_involved: string[];
  sources: Source[];
};

export type AuthResponse = {
  access_token: string;
  token_type: string;
  expires_at: string;
  user_id: string;
};

export type ChunkDetail = {
  id: string;
  text: string;
  document: string;
  page_num?: number;
  header_path?: string;
  [key: string]: unknown;
};
