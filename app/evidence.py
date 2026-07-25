"""教学 Agent 的结构化证据池。"""
import re
from dataclasses import dataclass, field


CHUNK_PATTERN = re.compile(
    r"\[Chunk\d+\s*\|\s*id=(?P<id>[^|\]]+)\s*\|\s*"
    r"rerank_score=(?P<score>\d+(?:\.\d+)?)\s*\|\s*第(?P<page>[^|\]]+)页"
)
GRAPH_PATTERN = re.compile(r"^\[(?P<type>数据结构|操作|复杂度)\]\s+(?P<name>.+)$", re.MULTILINE)


@dataclass
class EvidencePool:
    chunks: dict[str, dict] = field(default_factory=dict)
    graph_nodes: dict[str, dict] = field(default_factory=dict)

    @property
    def has_evidence(self) -> bool:
        return bool(self.chunks or self.graph_nodes)

    def add_observation(self, skill_name: str, observation: str) -> list[dict]:
        """解析一次 Skill 输出，返回本轮新增证据。"""
        added = []
        for match in CHUNK_PATTERN.finditer(observation):
            chunk_id = match.group("id").strip()
            page_text = match.group("page").strip()
            parsed_page = int(page_text) if page_text.isdigit() else page_text
            if isinstance(parsed_page, int) and parsed_page <= 0:
                parsed_page = None
            source = {
                "kind": "chunk",
                "id": chunk_id,
                "document": chunk_id.rsplit("_chunk_", 1)[0],
                "page_num": parsed_page,
                "rerank_score": float(match.group("score")),
                "skill": skill_name,
            }
            if chunk_id not in self.chunks:
                self.chunks[chunk_id] = source
                added.append(source)

        graph_match = GRAPH_PATTERN.search(observation)
        if graph_match:
            name = graph_match.group("name").strip()
            source = {
                "kind": "graph",
                "id": name,
                "node_type": graph_match.group("type"),
                "skill": skill_name,
            }
            if name not in self.graph_nodes:
                self.graph_nodes[name] = source
                added.append(source)
        return added

    def to_sources(self) -> list[dict]:
        return [*self.chunks.values(), *self.graph_nodes.values()]

    def select_for_answer(self, answer: str, limit: int = 3) -> list[dict]:
        """只保留正文实际提到的证据；未显式引用时回退到最强的少量证据。"""
        all_sources = self.to_sources()
        mentioned = [source for source in all_sources if source["id"] in answer]
        if mentioned:
            # 图谱节点是代码/复杂度路由的核心证据，即使正文只显式点名了教材
            # chunk，也应保留实际查询过的图谱节点以便复盘。
            graph_sources = list(self.graph_nodes.values())
            selected = []
            for source in [*graph_sources, *mentioned]:
                if source not in selected:
                    selected.append(source)
            if graph_sources and len(selected) < limit:
                ranked_chunks = sorted(
                    self.chunks.values(),
                    key=lambda source: float(source.get("rerank_score", 0)),
                    reverse=True,
                )
                for source in ranked_chunks:
                    if source not in selected:
                        selected.append(source)
                    if len(selected) >= limit:
                        break
            return selected[:limit]

        graph_sources = list(self.graph_nodes.values())
        ranked_chunks = sorted(
            self.chunks.values(),
            key=lambda source: float(source.get("rerank_score", 0)),
            reverse=True,
        )
        # 图谱查询通常承载代码和复杂度等核心结构化事实，优先保留。
        return [*graph_sources, *ranked_chunks][:limit]

    @staticmethod
    def reference_section(sources: list[dict]) -> str:
        lines = []
        for source in sources:
            if source["kind"] == "chunk":
                page = source.get("page_num")
                suffix = f"（第{page}页）" if page else "（页码未标注）"
                lines.append(f"- `{source['id']}`{suffix}")
            else:
                lines.append(f"- 图谱节点：{source['node_type']}「{source['id']}」")
        return "## 参考来源\n\n" + "\n".join(lines) if lines else ""


def append_references(answer: str, sources: list[dict] | EvidencePool) -> str:
    if isinstance(sources, EvidencePool):
        sources = sources.select_for_answer(answer)
    section = EvidencePool.reference_section(sources)
    if not section:
        return answer
    # 丢弃模型自行生成的来源列表，统一使用代码从真实证据池生成。
    answer_without_sources = answer.split("## 参考来源", 1)[0].rstrip()
    return f"{answer_without_sources}\n\n{section}"
