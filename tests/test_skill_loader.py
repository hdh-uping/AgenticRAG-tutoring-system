from app import skill_loader
from app.skill_loader import (
    build_skill_prompt,
    execute_skill,
    load_all_skills,
    load_skill_instructions,
)


def test_loads_three_skills_and_hides_post_answer_skill():
    skills = load_all_skills()
    assert {skill["name"] for skill in skills} == {
        "graph_lookup",
        "hybrid_retrieval",
        "related_concepts",
    }
    prompt = build_skill_prompt(skills)
    assert "hybrid_retrieval" in prompt
    assert "graph_lookup" in prompt
    assert "related_concepts" not in prompt
    assert all("module" not in skill and "body" not in skill for skill in skills)
    assert all(skill.get("skill_path") for skill in skills)


def test_skill_body_and_script_are_loaded_only_after_trigger(tmp_path, monkeypatch):
    skill_dir = tmp_path / "demo-skill"
    script_dir = skill_dir / "scripts"
    script_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: 仅用于测试懒加载。\n---\n\n"
        "# Demo\n\n触发后才应读取这段正文。\n",
        encoding="utf-8",
    )
    (script_dir / "run.py").write_text(
        "def run(arg):\n    return f'ran:{arg}'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skill_loader, "SKILLS_DIR", tmp_path)
    skill_loader._SKILL_MODULE_CACHE.clear()

    skills = load_all_skills()
    assert skills[0]["description"] == "仅用于测试懒加载。"
    assert "body" not in skills[0] and "module" not in skills[0]
    assert skill_loader._SKILL_MODULE_CACHE == {}

    instructions = load_skill_instructions("demo-skill", skills)
    assert "触发后才应读取" in instructions
    assert execute_skill("demo-skill", "ok", skills) == "ran:ok"
    assert len(skill_loader._SKILL_MODULE_CACHE) == 1
