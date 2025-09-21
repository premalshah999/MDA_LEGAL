import { useCallback, useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import { SiteHeader } from "@/components/layout/SiteHeader";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { ChatPanel } from "@/components/regnav/ChatPanel";
import { ChatSessionDetail, ChatSessionSummary, SearchFilters } from "@shared/api";
import { createDefaultSearchFilters } from "@/lib/searchFilters";
import { ensureUserId } from "@/lib/user";
import { toast } from "@/hooks/use-toast";
import { SourcesPanel } from "@/components/regnav/SourcesPanel";

export default function Chat() {
  const [userId, setUserId] = useState<string>("");
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [selectedSession, setSelectedSession] = useState<ChatSessionDetail | null>(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [filters, setFilters] = useState<SearchFilters>(() => createDefaultSearchFilters());
  const location = useLocation();

  useEffect(() => {
    if (typeof window === "undefined") return;
    setUserId(ensureUserId());
  }, []);

  const headers = useMemo(() => ({ "X-User-Id": userId, "Content-Type": "application/json" }), [userId]);

  const loadSessions = useCallback(async () => {
    if (!userId) return;
    setSessionsLoading(true);
    try {
      const res = await fetch("/api/chat/sessions", { headers });
      if (!res.ok) {
        throw new Error(`Failed to load sessions (${res.status})`);
      }
      const data = (await res.json()) as { sessions: ChatSessionSummary[] };
      setSessions(data.sessions ?? []);
    } catch (error) {
      const description = error instanceof Error ? error.message : "Could not load chat sessions.";
      toast({ title: "Error", description });
    } finally {
      setSessionsLoading(false);
    }
  }, [headers, userId]);

  const loadSessionDetail = useCallback(
    async (sessionId: string | null) => {
      if (!sessionId) {
        setSelectedSession(null);
        return;
      }
      setChatLoading(true);
      try {
        const res = await fetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, { headers });
        const payload = (await res.json()) as { session: ChatSessionDetail; error?: string };
        if (!res.ok) {
          throw new Error(payload.error ?? `Failed to load session (${res.status})`);
        }
        setSelectedSession(payload.session);
      } catch (error) {
        const description = error instanceof Error ? error.message : "Could not load session.";
        toast({ title: "Error", description });
        setSelectedSession(null);
      } finally {
        setChatLoading(false);
      }
    },
    [headers],
  );

  useEffect(() => {
    if (!userId) return;
    void loadSessions();
  }, [userId, loadSessions]);

  useEffect(() => {
    if (!userId) return;
    const params = new URLSearchParams(location.search);
    const sessionId = params.get("session");
    if (sessionId) {
      void loadSessionDetail(sessionId);
    }
  }, [location.search, loadSessionDetail, userId]);

  const handleSessionUpdated = useCallback(
    async (sessionId: string) => {
      await loadSessions();
      if (sessionId) {
        await loadSessionDetail(sessionId);
      }
    },
    [loadSessions, loadSessionDetail],
  );

  const handleSessionCreated = useCallback(
    async (sessionId: string) => {
      await loadSessions();
      await loadSessionDetail(sessionId);
    },
    [loadSessions, loadSessionDetail],
  );

  const resetSession = useCallback(() => {
    setSelectedSession(null);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6 lg:px-8">
        <section className="grid grid-cols-1 gap-6 md:grid-cols-12">
          {/* Main chat area */}
          <div className="md:col-span-8">
            <div className="rounded-2xl border bg-card p-4">
              <ChatPanel
                key={selectedSession?.id || "new"}
                filters={filters}
                userId={userId}
                session={selectedSession}
                loading={chatLoading}
                onSessionCreated={handleSessionCreated}
                onSessionUpdated={handleSessionUpdated}
                onResetSession={resetSession}
              />
            </div>
          </div>

          {/* Single sidebar: citations only */}
          <aside className="md:col-span-4">
            <div className="space-y-4">
              <div className="rounded-2xl border bg-card p-4">
                <SourcesPanel latestHits={selectedSession?.messages?.slice().reverse().find(m => m.role === 'assistant')?.citations ?? []} />
              </div>
            </div>
          </aside>
        </section>
      </main>

      <SiteFooter />
    </div>
  );
}
