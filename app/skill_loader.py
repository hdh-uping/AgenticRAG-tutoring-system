"""
Skill Loader — 以渐进式披露方式发现和执行项目 Skill。

发现阶段只读取 SKILL.md 的 name/description；Skill 首次触发时才读取正文并
动态导入 scripts/run.py，避免把未使用 Skill 的细节和实现提前加载。
"""
import importlib.util
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SKILLS_DIR = PROJECT_DIR / "skills"
_SKILL_MODULE_CACHE: dict[Path, object] = {}


def _parse_frontmatter(md_text: str) -> dict:
    """从 SKILL.md 中提取 YAML frontmatter。"""
    m = re.match(r"^---\s*\n(.*?)\n---", md_text, re.DOTALL)
    if not m:
        return {}
    meta = {}
    for line in m.group(1).split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def _read_skill_metadata(skill_md: Path) -> dict:
    """只读取文件开头的 frontmatter，不提前读取 Skill 正文。"""
    lines = []
    delimiter_count = 0
    with skill_md.open(encoding="utf-8") as handle:
        for line in handle:
            lines.append(line)
            if line.strip() == "---":
                delimiter_count += 1
                if delimiter_count == 2:
                    break
    return _parse_frontmatter("".join(lines))


def _load_skill_module(script_path: Path):
    """首次执行时动态加载 scripts/run.py，后续复用缓存模块。"""
    script_path = script_path.resolve()
    if script_path in _SKILL_MODULE_CACHE:
        return _SKILL_MODULE_CACHE[script_path]
    name = f"skill_{script_path.parent.parent.name}"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 Skill 脚本: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _SKILL_MODULE_CACHE[script_path] = mod
    return mod


def load_all_skills() -> list[dict]:
    """
    扫描 skills/*/SKILL.md，仅登记每个 Skill 的轻量元数据和资源路径。

    正文与 Python 模块均不在此阶段加载，确保发现阶段只披露 name/description。
    """
    skills = []
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        meta = _read_skill_metadata(skill_md)

        run_py = skill_dir / "scripts" / "run.py"

        # 判断是否在 Agent 循环内调用（vs 系统串行调用）
        in_agent_loop = meta.get("name") != "related_concepts"

        skills.append({
            "name": meta.get("name", skill_dir.name),
            "description": meta.get("description", ""),
            "skill_path": skill_md,
            "script_path": run_py if run_py.exists() else None,
            "in_agent_loop": in_agent_loop,
        })
    return skills


def load_skill_instructions(name: str, skills: list[dict]) -> str:
    """Skill 触发后才读取其 SKILL.md 正文，供 Agent 理解结果和后续决策。"""
    for skill in skills:
        if skill.get("name") != name:
            continue
        skill_path = skill.get("skill_path")
        if not skill_path:
            return ""
        md_text = Path(skill_path).read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n.*?\n---", md_text, re.DOTALL)
        return md_text[match.end():].strip() if match else md_text.strip()
    return ""


def build_skill_prompt(skills: list[dict]) -> str:
    """为 Agent 的 system prompt 生成工具描述段落。"""
    lines = ["【你可以调用的技能】"]
    for s in skills:
        if s.get("in_agent_loop", True):
            lines.append(f"- {s['name']}: {s['description']}")
    return "\n".join(lines)


def execute_skill(name: str, arg: str, skills: list[dict]) -> str:
    """
    Agent 调用 Skill 的入口。

    Args:
        name: Skill 名（如 hybrid_retrieval）
        arg:  参数（如查询字符串或概念名）
        skills: load_all_skills() 的返回值

    Returns:
        Skill 执行结果（文本）
    """
    for s in skills:
        if s.get("name") == name:
            script_path = s.get("script_path")
            if not script_path:
                return f"Skill 缺少可执行脚本: {name}"
            module = _load_skill_module(Path(script_path))
            # related_concepts 接受逗号分隔的概念列表
            if name == "related_concepts":
                concepts = [c.strip() for c in arg.split(",") if c.strip()]
                return module.run(concepts)
            else:
                return module.run(arg)
    return f"未知技能: {name}"
