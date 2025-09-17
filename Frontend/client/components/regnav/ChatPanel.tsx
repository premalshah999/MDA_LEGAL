import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MessageBubble } from "./MessageBubble";
import { ScoredDoc, SearchFilters, SearchRequest, SearchResponse } from "@shared/api";
import { Loader2, Send, Save } from "lucide-react";
import { toast } from "@/hooks/use-toast";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: ScoredDoc[];
}

export function ChatPanel({ filters, initialMessages }: { filters: SearchFilters; initialMessages?: Message[] }) {
  const welcome: Message = {
    id: "welcome",
    role: "assistant",
    content: "Ask about Maryland Department of Agriculture/Environment regulations. Use the filters to target jurisdiction, agency, or year.",
  };

  const [messages, setMessages] = useState<Message[]>(initialMessages && initialMessages.length > 0 ? initialMessages : [welcome]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (initialMessages) setMessages(initialMessages);
  }, [initialMessages]);

  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    const userMsg: Message = { id: crypto.randomUUID(), role: "user", content: q };
    setMessages((m) => [...m, userMsg]);
    setLoading(true);
    try {
      const payload: SearchRequest = { query: q, filters, topK: 4 };
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = (await res.json()) as SearchResponse;
      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer,
        citations: data.hits,
      };
      window.dispatchEvent(new CustomEvent("regnav:citations", { detail: data.hits }));
      setMessages((m) => [...m, assistantMsg]);
    } catch (e) {
      const err: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "There was an error processing your query.",
      };
      setMessages((m) => [...m, err]);
      toast({ title: "Search error", description: "Could not run the query. Try again." });
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

  const saveSession = () => {
    const title = window.prompt("Save chat as", "New chat");
    if (!title) return;

    try {
      const sessions = JSON.parse(localStorage.getItem("regnav:sessions") || "[]");
      const snippet = messages.find((m) => m.role === "assistant")?.content.slice(0, 120) || messages[0]?.content.slice(0, 120) || "";
      const session = { id: crypto.randomUUID(), title, snippet, time: new Date().toISOString(), messages };
      sessions.unshift(session);
      localStorage.setItem("regnav:sessions", JSON.stringify(sessions));
      window.dispatchEvent(new CustomEvent("regnav:sessions:updated"));
      toast({ title: "Saved", description: "Chat saved to history." });
    } catch {
      toast({ title: "Error", description: "Could not save chat." });
    }
  };

  const newChat = () => setMessages([welcome]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-4 rounded-t-xl border-b px-3 py-2">
        <div>
          <div className="text-sm font-semibold">Workspace</div>
          <div className="text-xs text-muted-foreground">Interactive regulatory QA</div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" onClick={newChat} size="sm">New</Button>
          <Button variant="outline" onClick={saveSession} size="sm"><Save className="h-4 w-4" /></Button>
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-4 pr-2">
        {messages.map((m) => (
          <div key={m.id} className="space-y-2">
            <MessageBubble role={m.role}>{m.content}</MessageBubble>
            {m.citations && m.citations.length > 0 && (
              <div className="ml-8 mt-1 text-xs text-muted-foreground">
                <div className="font-semibold">Citations:</div>
                <ul className="list-disc list-inside">
                  {m.citations.map((c) => (
                    <li key={c.meta.id}>{c.meta.title ?? c.meta.id}</li>
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
        <Button onClick={send} disabled={loading || !input.trim()}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          <span className="ml-2 hidden sm:inline">Send</span>
        </Button>
      </div>
    </div>
  );
}
