import { useCallback, useEffect, useMemo, useState } from "react";
import { SiteHeader } from "@/components/layout/SiteHeader";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { MessageBubble } from "@/components/regnav/MessageBubble";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { ChatMemory, ChatSessionDetail, ChatSessionSummary } from "@shared/api";
import { ensureUserId } from "@/lib/user";
import { toast } from "@/hooks/use-toast";

export default function Admin() {
  const [userId, setUserId] = useState<string>("");
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [selected, setSelected] = useState<ChatSessionDetail | null>(null);
  const [memories, setMemories] = useState<ChatMemory[]>([]);
  const [memoriesLoading, setMemoriesLoading] = useState(false);
  const [newMemory, setNewMemory] = useState("");
  const navigate = useNavigate();

  const headers = useMemo(() => ({ "X-User-Id": userId, "Content-Type": "application/json" }), [userId]);

  const loadSessions = useCallback(async () => {
    if (!userId) return;
    setSessionsLoading(true);
    try {
      const res = await fetch("/api/chat/sessions", { headers });
      const payload = (await res.json()) as { sessions: ChatSessionSummary[]; error?: string };
      if (!res.ok) throw new Error(payload.error ?? `Failed to load sessions (${res.status})`);
      setSessions(payload.sessions ?? []);
    } catch (error) {
      const description = error instanceof Error ? error.message : "Could not load sessions.";
      toast({ title: "Error", description });
    } finally {
      setSessionsLoading(false);
    }
  }, [headers, userId]);

  const loadSessionDetail = useCallback(
    async (sessionId: string) => {
      if (!userId) return;
      try {
        const res = await fetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, { headers });
        const payload = (await res.json()) as { session: ChatSessionDetail; error?: string };
        if (!res.ok) throw new Error(payload.error ?? `Failed to load session (${res.status})`);
        setSelected(payload.session);
      } catch (error) {
        const description = error instanceof Error ? error.message : "Could not load session.";
        toast({ title: "Error", description });
        setSelected(null);
      }
    },
    [headers, userId],
  );

  const loadMemories = useCallback(async () => {
    if (!userId) return;
    setMemoriesLoading(true);
    try {
      const res = await fetch("/api/chat/memories", { headers });
      const payload = (await res.json()) as { memories: ChatMemory[]; error?: string };
      if (!res.ok) throw new Error(payload.error ?? `Failed to load memories (${res.status})`);
      setMemories(payload.memories ?? []);
    } catch (error) {
      const description = error instanceof Error ? error.message : "Could not load memories.";
      toast({ title: "Error", description });
    } finally {
      setMemoriesLoading(false);
    }
  }, [headers, userId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const id = ensureUserId();
    setUserId(id);
  }, []);

  useEffect(() => {
    if (!userId) return;
    void loadSessions();
    void loadMemories();
  }, [loadMemories, loadSessions, userId]);

  const refresh = useCallback(() => {
    void loadSessions();
    void loadMemories();
  }, [loadMemories, loadSessions]);

  const remove = useCallback(
    async (id: string) => {
      try {
        const res = await fetch(`/api/chat/sessions/${encodeURIComponent(id)}`, {
          method: "DELETE",
          headers,
        });
        if (!res.ok) {
          const payload = (await res.json()) as { error?: string };
          throw new Error(payload.error ?? `Failed to delete session (${res.status})`);
        }
        if (selected?.id === id) setSelected(null);
        toast({ title: "Deleted", description: "Session removed." });
        await loadSessions();
      } catch (error) {
        const description = error instanceof Error ? error.message : "Could not delete session.";
        toast({ title: "Error", description });
      }
    },
    [headers, loadSessions, selected?.id],
  );

  const clearAll = useCallback(async () => {
    for (const session of sessions) {
      await remove(session.id);
    }
    setSelected(null);
  }, [remove, sessions]);

  const exportAll = () => {
    const blob = new Blob([JSON.stringify({ sessions, memories }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "regnav-sessions.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const openInChat = (id: string) => {
    navigate(`/chat?session=${encodeURIComponent(id)}`);
  };

  const addNewMemory = useCallback(async () => {
    const content = newMemory.trim();
    if (!content) return;
    try {
      const res = await fetch("/api/chat/memories", {
        method: "POST",
        headers,
        body: JSON.stringify({ content }),
      });
      const payload = (await res.json()) as { memory: ChatMemory; error?: string };
      if (!res.ok) throw new Error(payload.error ?? `Failed to add memory (${res.status})`);
      setNewMemory("");
      toast({ title: "Saved", description: "Memory added." });
      await loadMemories();
    } catch (error) {
      const description = error instanceof Error ? error.message : "Could not add memory.";
      toast({ title: "Error", description });
    }
  }, [headers, loadMemories, newMemory]);

  const removeMemory = useCallback(
    async (id: string) => {
      try {
        const res = await fetch(`/api/chat/memories/${encodeURIComponent(id)}`, {
          method: "DELETE",
          headers,
        });
        if (!res.ok) {
          const payload = (await res.json()) as { error?: string };
          throw new Error(payload.error ?? `Failed to delete memory (${res.status})`);
        }
        toast({ title: "Deleted", description: "Memory removed." });
        await loadMemories();
      } catch (error) {
        const description = error instanceof Error ? error.message : "Could not delete memory.";
        toast({ title: "Error", description });
      }
    },
    [headers, loadMemories],
  );

  const handleSessionSelect = (sessionId: string) => {
    void loadSessionDetail(sessionId);
  };

  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-12">
          <aside className="md:col-span-4">
            <div className="rounded-2xl border bg-card p-4 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Admin — Sessions</h2>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={refresh}>Refresh</Button>
                  <Button variant="ghost" size="sm" onClick={exportAll}>Export</Button>
                  <Button variant="destructive" size="sm" onClick={clearAll}>Clear all</Button>
                </div>
              </div>

              {sessionsLoading ? (
                <div className="text-sm text-muted-foreground">Loading sessions…</div>
              ) : sessions.length === 0 ? (
                <div className="text-sm text-muted-foreground">No sessions found.</div>
              ) : (
                <div className="divide-y">
                  {sessions.map((s) => (
                    <div key={s.id} className="py-3 flex items-start justify-between">
                      <div className="cursor-pointer" onClick={() => handleSessionSelect(s.id)}>
                        <div className="font-semibold">{s.title}</div>
                        {s.snippet && <div className="text-xs text-muted-foreground">{s.snippet}</div>}
                        <div className="text-xs text-muted-foreground">{new Date(s.updatedAt).toLocaleString()}</div>
                      </div>
                      <div className="flex flex-col items-end gap-2">
                        <div className="flex gap-2">
                          <Button size="sm" variant="ghost" onClick={() => openInChat(s.id)}>Open</Button>
                          <Button size="sm" variant="outline" onClick={() => remove(s.id)}>Delete</Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </aside>

          <div className="md:col-span-8">
            <div className="rounded-2xl border bg-card p-4">
              <h2 className="text-lg font-semibold mb-4">Session preview</h2>
              {!selected ? (
                <div className="text-sm text-muted-foreground">Select a session to preview its messages.</div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-semibold">{selected.title}</div>
                      <div className="text-xs text-muted-foreground">{new Date(selected.updatedAt).toLocaleString()}</div>
                    </div>
                    <div className="flex gap-2">
                      <Button onClick={() => openInChat(selected.id)}>Open in Chat</Button>
                      <Button variant="outline" onClick={() => remove(selected.id)}>Delete</Button>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {selected.messages.map((m) => (
                      <div key={m.id}>
                        <MessageBubble role={m.role}>{m.content}</MessageBubble>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="md:col-span-4">
            <div className="rounded-2xl border bg-card p-4 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">User memory</h2>
                <Button variant="outline" size="sm" onClick={loadMemories}>
                  Refresh
                </Button>
              </div>

              <div className="space-y-3">
                <textarea
                  className="w-full rounded-md border bg-background p-2 text-sm"
                  rows={3}
                  placeholder="Add a new memory about the user…"
                  value={newMemory}
                  onChange={(e) => setNewMemory(e.target.value)}
                />
                <Button onClick={addNewMemory} disabled={!newMemory.trim()}>
                  Save memory
                </Button>
              </div>

              {memoriesLoading ? (
                <div className="text-sm text-muted-foreground">Loading memories…</div>
              ) : memories.length === 0 ? (
                <div className="text-sm text-muted-foreground">No persistent memories captured yet.</div>
              ) : (
                <div className="space-y-3">
                  {memories.map((memory) => (
                    <div key={memory.id} className="rounded-md border p-3 text-sm">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <div>{memory.content}</div>
                          <div className="text-xs text-muted-foreground">Updated {new Date(memory.updatedAt).toLocaleString()}</div>
                        </div>
                        <Button variant="ghost" size="sm" onClick={() => removeMemory(memory.id)}>
                          Delete
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <SiteFooter />
    </div>
  );
}
