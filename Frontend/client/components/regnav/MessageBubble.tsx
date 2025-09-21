import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";

export function MessageBubble({ role, content }: { role: "user" | "assistant"; content: string }) {
  const isUser = role === "user";
  const bubble = isUser ? "bg-primary text-primary-foreground" : "bg-card text-foreground";
  const align = isUser ? "justify-end" : "justify-start";

  return (
    <div className={cn("flex w-full items-start gap-3", align)}>
      {!isUser && (
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted text-sm font-medium text-muted-foreground">RN</div>
      )}

      <div className={cn("max-w-[80%] rounded-2xl px-4 py-3 text-sm shadow-md", bubble)}>
        {isUser ? (
          <div>{content}</div>
        ) : (
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <ReactMarkdown
              components={{
                h1: ({ children }) => <h1 className="text-lg font-bold mb-3 mt-4 first:mt-0">{children}</h1>,
                h2: ({ children }) => <h2 className="text-base font-bold mb-2 mt-3 first:mt-0">{children}</h2>,
                h3: ({ children }) => <h3 className="text-sm font-bold mb-2 mt-2 first:mt-0">{children}</h3>,
                p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>,
                ul: ({ children }) => <ul className="mb-3 pl-4 space-y-1 list-disc">{children}</ul>,
                ol: ({ children }) => <ol className="mb-3 pl-4 space-y-1 list-decimal">{children}</ol>,
                li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                strong: ({ children }) => <strong className="font-bold">{children}</strong>,
                em: ({ children }) => <em className="italic">{children}</em>,
                code: ({ children }) => <code className="bg-muted px-1 py-0.5 rounded text-xs">{children}</code>,
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        )}
      </div>

      {isUser && (
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20 text-primary text-sm font-medium">U</div>
      )}
    </div>
  );
}
