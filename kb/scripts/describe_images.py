"""
用本地 Qwen3-VL-4B 视觉模型将 Markdown 中的图片替换为文字描述，
保证纯文本嵌入时语义完整。

用法:
    python build_kb/describe_images.py

依赖:
    pip install transformers torch accelerate pillow qwen-vl-utils
"""

import argparse
import re
import sys
import os
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from dotenv import load_dotenv

# ── 配置 ────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")
PROCESSED_DIR = BASE_DIR / "kb" / "processed"
MODEL_PATH = os.getenv(
    "VLM_MODEL_PATH",
    str(BASE_DIR.parent / "models" / "Qwen" / "Qwen3-VL-4B-Instruct"),
)

# 图片描述 prompt（中文，适配教材内容）
DESCRIBE_PROMPT = (
    "请用1-2句简洁的中文描述这张图片的内容。"
    "这是一本中文数据结构教材中的插图，可能包含数据结构示意图、算法流程图、"
    "表格或公式。描述要具体、准确，直接说明图中展示了什么，不要加任何前缀。"
)

# ── 核心逻辑 ────────────────────────────────────────────────────

def find_image_refs(md_text: str) -> list[tuple[str, str]]:
    """扫描 md 文本，返回所有图片引用。Returns: [(完整匹配串, 图片路径), ...]"""
    pattern = re.compile(r'!\[\]\((images/[^)]+)\)')
    return [(m.group(0), m.group(1)) for m in pattern.finditer(md_text)]


def load_model(model_path: str):
    """加载本地 Qwen3-VL 模型（仅加载一次）。"""
    print(f"⏳ 加载模型: {model_path} ...")

    # Qwen3-VL 在 MPS 下会 segfault，强制用 CPU
    # 4B 模型在 Apple Silicon CPU 上推理约 10-20 秒/张，24 张图可接受
    device = "cpu"
    print("   🖥️  使用 CPU（MPS 与 VL 模型不兼容，已自动切换）")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,       # 半精度 ~8GB，16G 内存刚好
        device_map="cpu",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"   ✅ 模型加载完成")
    return model, processor, device


def describe_image(
    model, processor, image_path: Path, device: str
) -> str:
    """用本地 Qwen3-VL 描述单张图片。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": DESCRIBE_PROMPT},
            ],
        }
    ]

    # 用 qwen_vl_utils 处理视觉信息
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,         # 确定性输出
        )

    # 去掉输入部分，只保留生成的新 token
    input_len = inputs.input_ids.shape[1]
    output_ids = generated_ids[0][input_len:]
    result = processor.decode(output_ids, skip_special_tokens=True).strip()

    return result


def replace_images_in_md(
    md_text: str,
    image_dir: Path,
    model, processor, device: str,
) -> str:
    """逐张替换 md 中的图片为纯文本描述。"""
    refs = find_image_refs(md_text)
    print(f"🔍 发现 {len(refs)} 处图片引用\n")

    cache: dict[str, str] = {}

    for i, (full_match, img_rel_path) in enumerate(refs, 1):
        img_name = Path(img_rel_path).name
        img_path = image_dir / img_name

        if not img_path.exists():
            print(f"  [{i}/{len(refs)}] ⚠️  文件不存在: {img_name} → 跳过")
            continue

        if img_name in cache:
            desc = cache[img_name]
            print(f"  [{i}/{len(refs)}] 📎 {img_name} → (缓存)")
        else:
            print(f"  [{i}/{len(refs)}] 🔄 {img_name} → 本地模型推理...")
            try:
                desc = describe_image(model, processor, img_path, device)
                cache[img_name] = desc
                print(f"         → {desc[:80]}{'...' if len(desc) > 80 else ''}")
            except Exception as e:
                print(f"         ❌ 失败: {e}")
                continue

        # 纯文本替换，不加任何装饰符号
        md_text = md_text.replace(full_match, desc, 1)

    return md_text


# ── 主流程 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", help="只处理指定文档，例如：数组")
    parser.add_argument("--md-file", type=Path, help="直接处理指定 MinerU Markdown")
    args = parser.parse_args()

    # 1. 找到 processed 下的 md 文件
    if args.md_file:
        md_files = [args.md_file.resolve()]
    else:
        md_files = list(PROCESSED_DIR.glob("*/*.md"))
        md_files = [f for f in md_files if not f.stem.endswith("_text")]
    if args.doc and not args.md_file:
        md_files = [
            f for f in md_files
            if f.parent.name == args.doc and f.stem == args.doc
        ]

    if not md_files:
        print("❌ 未找到待处理的 md 文件")
        sys.exit(1)

    print(f"📂 找到 {len(md_files)} 个 md 文件:")
    for f in md_files:
        print(f"   - {f.relative_to(BASE_DIR)}")

    # 2. 仅在确有图片需要描述时加载本地模型
    pending_images = False
    for md_file in md_files:
        md_text = md_file.read_text(encoding="utf-8")
        pending_images = pending_images or bool(find_image_refs(md_text))

    model = processor = device = None
    if pending_images:
        model, processor, device = load_model(MODEL_PATH)

    # 3. 逐个文件处理
    for md_file in md_files:
        print(f"\n{'='*60}")
        print(f"📄 处理: {md_file.name}")

        md_text = md_file.read_text(encoding="utf-8")
        image_dir = md_file.parent / "images"

        refs = find_image_refs(md_text)
        if refs and not image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {image_dir}")

        new_text = (
            replace_images_in_md(md_text, image_dir, model, processor, device)
            if refs else md_text
        )

        # 输出为 xxx_text.md
        output_path = md_file.parent / f"{md_file.stem}_text.md"
        output_path.write_text(new_text, encoding="utf-8")
        print(f"\n✅ 已保存: {output_path.relative_to(BASE_DIR)}")

    print("\n" + "=" * 60)
    print("🎉 处理完成")


if __name__ == "__main__":
    main()
