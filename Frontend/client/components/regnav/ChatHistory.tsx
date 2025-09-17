import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";

export function ChatHistory({ onSelect }: { onSelect?: (s: any) => void }) {
  const [sessions, setSessions] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("regnav:sessions") || "[]");
    } catch {
      return [];
    }
  });

  useEffect(() => {
    const handler = () => {
      try {
        setSessions(JSON.parse(localStorage.getItem("regnav:sessions") || "[]"));
      } catch {
        setSessions([]);
      }
    };
    window.addEventListener("regnav:sessions:updated", handler as EventListener);
    return () => window.removeEventListener("regnav:sessions:updated", handler as EventListener);
  }, []);

  const newChat = () => {
    if (onSelect) onSelect(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const remove = (id: string) => {
    const updated = sessions.filter((s: any) => s.id !== id);
    setSessions(updated);
    localStorage.setItem("regnav:sessions", JSON.stringify(updated));
    window.dispatchEvent(new CustomEvent("regnav:sessions:updated"));
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium">Recent conversations</div>
        <Button variant="outline" size="sm" onClick={newChat}>New chat</Button>
      </div>

      <div className="divide-y">
        {sessions.length === 0 ? (
          <div className="py-3 text-sm text-muted-foreground">No saved chats yet. Start a new conversation and save it.</div>
        ) : (
          sessions.map((s: any) => (
            <div key={s.id} className="py-3">
              <div className="flex items-center justify-between">
                <div className="cursor-pointer" onClick={() => onSelect && onSelect(s)}>
                  <div className="font-semibold">{s.title}</div>
                  <div className="text-xs text-muted-foreground">{s.snippet}</div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="text-xs text-muted-foreground">{new Date(s.time).toLocaleString()}</div>
                  <Button variant="ghost" size="sm" onClick={() => remove(s.id)}>Delete</Button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
