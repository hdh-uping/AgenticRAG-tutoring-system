function convertMathOutsideInlineCode(value: string): string {
  const inlineCode = /(`+)([\s\S]*?)\1/g;
  let normalized = "";
  let cursor = 0;

  function convert(valuePart: string) {
    return valuePart
      .replace(/(?<!\\)\\\[([\s\S]*?)(?<!\\)\\\]/g, (_, expression: string) => (
        `\n\n$$\n${expression.trim()}\n$$\n\n`
      ))
      .replace(/(?<!\\)\\\(([\s\S]*?)(?<!\\)\\\)/g, (_, expression: string) => (
        `$${expression.trim()}$`
      ));
  }

  for (const match of value.matchAll(inlineCode)) {
    const index = match.index ?? 0;
    normalized += convert(value.slice(cursor, index));
    normalized += match[0];
    cursor = index + match[0].length;
  }
  normalized += convert(value.slice(cursor));
  return normalized;
}

/**
 * remark-math 识别 $...$ / $$...$$，但教学模型也经常返回 \(...\) / \[...\]。
 * 转换普通 Markdown 文本中的 LaTeX 分隔符，同时保持代码块和行内代码原样。
 */
export function normalizeMathDelimiters(markdown: string): string {
  const lines = markdown.match(/.*(?:\n|$)/g)?.filter(Boolean) ?? [];
  const output: string[] = [];
  let plainText = "";
  let fence: { marker: string; length: number } | null = null;

  function flushPlainText() {
    if (!plainText) return;
    output.push(convertMathOutsideInlineCode(plainText));
    plainText = "";
  }

  for (const line of lines) {
    if (!fence) {
      const opening = line.match(/^[ \t]{0,3}(`{3,}|~{3,})/);
      if (!opening) {
        plainText += line;
        continue;
      }
      flushPlainText();
      fence = { marker: opening[1][0], length: opening[1].length };
      output.push(line);
      continue;
    }

    output.push(line);
    const closingPattern = new RegExp(
      `^[ \\t]{0,3}${fence.marker}{${fence.length},}[ \\t]*(?:\\n|$)`,
    );
    if (closingPattern.test(line)) fence = null;
  }

  flushPlainText();
  return output.join("");
}
