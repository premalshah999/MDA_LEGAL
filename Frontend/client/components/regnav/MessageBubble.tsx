import { cn } from "@/lib/utils";

export function MessageBubble({ role, children }: { role: "user" | "assistant"; children: React.ReactNode }) {
  const isUser = role === "user";
  const bubble = isUser ? "bg-primary text-primary-foreground" : "bg-card text-foreground";
  const align = isUser ? "justify-end" : "justify-start";

  return (
    <div className={cn("flex w-full items-start gap-3", align)}>
      {!isUser && (
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted text-sm font-medium text-muted-foreground">RN</div>
      )}

      <div className={cn("max-w-[80%] rounded-2xl px-4 py-3 text-sm shadow-md", bubble)}>
        {children}
      </div>

      {isUser && (
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20 text-primary text-sm font-medium">U</div>
      )}
    </div>
  );
}
