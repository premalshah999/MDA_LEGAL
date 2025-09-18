import { Button } from "@/components/ui/button";
import { ChatSessionSummary } from "@shared/api";

interface ChatHistoryProps {
  sessions: ChatSessionSummary[];
  loading?: boolean;
  onSelect?: (sessionId: string | null) => void;
  onDelete?: (sessionId: string) => void;
}

export function ChatHistory({ sessions, loading, onSelect, onDelete }: ChatHistoryProps) {
  const startNewChat = () => {
    onSelect?.(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-medium">Recent conversations</div>
        <Button variant="outline" size="sm" onClick={startNewChat}>
          New chat
        </Button>
      </div>

      <div className="divide-y">
        {loading ? (
          <div className="py-3 text-sm text-muted-foreground">Loading sessions…</div>
        ) : sessions.length === 0 ? (
          <div className="py-3 text-sm text-muted-foreground">
            No saved chats yet. Start a new conversation to begin tracking history.
          </div>
        ) : (
          sessions.map((s) => (
            <div key={s.id} className="py-3">
              <div className="flex items-center justify-between">
                <button
                  type="button"
                  className="text-left"
                  onClick={() => onSelect?.(s.id)}
                >
                  <div className="font-semibold">{s.title}</div>
                  {s.snippet && <div className="text-xs text-muted-foreground">{s.snippet}</div>}
                  <div className="text-xs text-muted-foreground">{new Date(s.updatedAt).toLocaleString()}</div>
                </button>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onDelete?.(s.id)}
                  >
                    Delete
                  </Button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
