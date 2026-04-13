import type { ComponentPropsWithoutRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { MermaidBlock } from "./MermaidBlock";
import { stripLegacyEvidenceTags } from "../lib/evidence";

type Props = { content: string };

function Code({
  inline,
  className,
  children,
  ...props
}: ComponentPropsWithoutRef<"code"> & { inline?: boolean }) {
  const match = /language-(\w+)/.exec(className || "");
  const lang = match?.[1];
  const code = String(children).replace(/\n$/, "");
  if (!inline && lang === "mermaid") {
    return <MermaidBlock chart={code} />;
  }
  return (
    <code className={className} {...props}>
      {children}
    </code>
  );
}

export function MarkdownBody({ content }: Props) {
  const safe = stripLegacyEvidenceTags(content);
  return (
    <div className="prose prose-invert prose-sm max-w-none prose-headings:font-semibold prose-a:text-violet-400 prose-code:text-cyan-300 prose-pre:bg-zinc-950 prose-pre:border prose-pre:border-zinc-800 prose-table:text-sm prose-img:rounded-lg prose-img:border prose-img:border-zinc-800 prose-img:max-w-full prose-img:mx-auto">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ code: Code }}>
        {safe}
      </ReactMarkdown>
    </div>
  );
}
