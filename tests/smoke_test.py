"""手工烟雾测试。

该脚本会调用本地模型、Neo4j 和外部 LLM，不属于默认离线 pytest 套件。
运行方式：``python tests/smoke_test.py``。
"""
from app.db import DEFAULT_PREFS
from app.workflow import run_tutoring_workflow


QUESTIONS = [
    "顺序表怎么插入元素？",
    "单链表和顺序表的插入操作有什么区别？",
    "顺序表删除操作的 C 代码怎么写？",
]


def main():
    for question in QUESTIONS:
        print(f"\n{'=' * 70}\n问题: {question}\n{'=' * 70}")
        result = run_tutoring_workflow(question=question, prefs=DEFAULT_PREFS)
        print(f"轮数: {result['iterations']}")
        for item in result["trace"]:
            print(f"  [轮{item['turn']}] {item.get('action', '')} {item.get('input', '')}")
        print(f"\n答案:\n{result['answer']}")

        if result["recommendation"]:
            print(f"\n推荐 Agent 已生成关联学习建议。")


if __name__ == "__main__":
    main()
