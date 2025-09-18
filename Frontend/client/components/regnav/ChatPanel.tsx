import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MessageBubble } from "./MessageBubble";
import { ChatAskRequest, ChatAskResponse, ChatMessage, ChatSessionDetail, ChatCitation, SearchFilters } from "@shared/api";
import { Loader2, Send, Save } from "lucide-react";
import { toast } from "@/hooks/use-toast";

const WELCOME: ChatMessage = {
    id: "welcome",
    role: "assistant",
    content: "Ask about Maryland Department of Agriculture/Environment regulations. Use the filters to target jurisdiction, agency, or year.",
    createdAt: new Date().toISOString(),
  };

interface ChatPanelProps {
  filters: SearchFilters;
  userId: string;
  session?: ChatSessionDetail | null;
  loading?: boolean;
  onSessionCreated?: (sessionId: string) => void;
  onSessionUpdated?: (sessionId: string) => Promise<void> | void;
  onResetSession?: () => void;
}

export function ChatPanel({ filters, userId, session, loading: sessionLoading, onSessionCreated, onSessionUpdated, onResetSession }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(session?.messages?.length ? session.messages : [WELCOME]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(session?.id ?? null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (session?.messages?.length) {
      setMessages(session.messages);
    } else {
      setMessages([WELCOME]);
    }
    setActiveSessionId(session?.id ?? null);
  }, [session?.id, session?.messages]);

  const send = async () => {
    const q = input.trim();
    if (!q || loading || sessionLoading || !userId) return;
    setInput("");
    const placeholderId = crypto.randomUUID();
    const now = new Date().toISOString();
    const placeholder: ChatMessage = { id: placeholderId, role: "user", content: q, createdAt: now };
    setMessages((m) => [...m, placeholder]);
    setLoading(true);
    try {
      const payload: ChatAskRequest = {
        question: q,
        filters,
        topK: 4,
        sessionId: activeSessionId ?? undefined,
        sessionTitle: session?.title ?? q.slice(0, 60),
      };
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-User-Id": userId },
        body: JSON.stringify(payload),
      });

      let parsed: unknown = null;
      try {
        parsed = await res.json();
      } catch {
        // Ignore JSON parsing errors so we can surface a clearer toast message below.
      }

      if (!res.ok) {
        const errorMessage =
          (parsed && typeof parsed === "object" && "error" in parsed && typeof (parsed as any).error === "string"
            ? (parsed as any).error
            : undefined) ??
          `Request failed with status ${res.status}`;
        throw new Error(errorMessage);
      }

      if (!parsed || typeof parsed !== "object" || !("sessionId" in parsed)) {
        throw new Error("Malformed response from server");
      }

      const data = parsed as ChatAskResponse;
      const updatedSessionId = data.sessionId;
      const assistantCitations: ChatCitation[] = data.sources ?? [];
      const userMessage = data.userMessage;
      const assistantMessage: ChatMessage = {
        ...data.assistantMessage,
        citations: assistantCitations,
      };
      setMessages((m) => {
        const withoutPlaceholder = m.filter((msg) => msg.id !== placeholderId);
        return [...withoutPlaceholder, userMessage, assistantMessage];
      });
      setActiveSessionId(updatedSessionId);
      window.dispatchEvent(new CustomEvent("regnav:citations", { detail: assistantCitations }));
      if (!session?.id && !activeSessionId) {
        onSessionCreated?.(updatedSessionId);
      }
      await onSessionUpdated?.(updatedSessionId);
    } catch (e) {
      const description = e instanceof Error ? e.message : "Could not run the query. Try again.";
      const err: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "There was an error processing your query.",
        createdAt: new Date().toISOString(),
      };
      setMessages((m) => m.filter((msg) => msg.id !== placeholderId).concat(err));
      toast({ title: "Search error", description });
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      send();
    }
  };

  const renameSession = async () => {
    if (!activeSessionId || !userId) {
      toast({ title: "No session", description: "Send a message before saving the chat." });
      return;
    }
    const title = window.prompt("Save chat as", session?.title ?? "New chat");
    if (!title) return;
    try {
      const res = await fetch(`/api/chat/sessions/${encodeURIComponent(activeSessionId)}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          "X-User-Id": userId,
        },
        body: JSON.stringify({ title }),
      });
      if (!res.ok) {
        const { error } = (await res.json()) as { error?: string };
        throw new Error(error ?? `Rename failed (${res.status})`);
      }
      toast({ title: "Saved", description: "Chat title updated." });
      await onSessionUpdated?.(activeSessionId);
    } catch (error) {
      const description = error instanceof Error ? error.message : "Could not rename chat.";
      toast({ title: "Error", description });
    }
  };

  const newChat = () => {
    setMessages([WELCOME]);
    setActiveSessionId(null);
    onResetSession?.();
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-4 rounded-t-xl border-b px-3 py-2">
        <div>
          <div className="text-sm font-semibold">Workspace</div>
          <div className="text-xs text-muted-foreground">Interactive regulatory QA</div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" onClick={newChat} size="sm" disabled={loading || sessionLoading}>
            New
          </Button>
          <Button variant="outline" onClick={renameSession} size="sm" disabled={loading || sessionLoading}>
            <Save className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-4 pr-2">
        {(sessionLoading ? [
          { id: "loading", role: "assistant", content: "Loading session…", createdAt: new Date().toISOString() },
        ] : messages).map((m) => (
          <div key={m.id} className="space-y-2">
            <MessageBubble role={m.role}>{m.content}</MessageBubble>
            {m.citations && m.citations.length > 0 && (
              <div className="ml-8 mt-1 text-xs text-muted-foreground">
                <div className="font-semibold">Citations:</div>
                <ul className="list-disc list-inside">
                  {m.citations.map((c) => (
                    <li key={c.id}>{c.label}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div className="sticky bottom-0 mt-3 flex items-center gap-2 rounded-b-xl border-t bg-background/5 p-3">
        <div className="flex-1">
          <Input
            placeholder="Ask about regulations, e.g., 'What are poultry litter storage requirements in Maryland?'"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            className="w-full"
          />
        </div>
        <Button onClick={send} disabled={loading || sessionLoading || !input.trim() || !userId}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          <span className="ml-2 hidden sm:inline">Send</span>
        </Button>
      </div>
    </div>
  );
}
