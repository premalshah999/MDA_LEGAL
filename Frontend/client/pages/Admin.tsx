import { useEffect, useState } from "react";
import { SiteHeader } from "@/components/layout/SiteHeader";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { MessageBubble } from "@/components/regnav/MessageBubble";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

export default function Admin() {
  const [sessions, setSessions] = useState<any[]>([]);
  const [selected, setSelected] = useState<any | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    try {
      const s = JSON.parse(localStorage.getItem("regnav:sessions") || "[]");
      setSessions(s);
    } catch {
      setSessions([]);
    }
  }, []);

  const refresh = () => {
    try {
      const s = JSON.parse(localStorage.getItem("regnav:sessions") || "[]");
      setSessions(s);
    } catch {
      setSessions([]);
    }
  };

  const remove = (id: string) => {
    const updated = sessions.filter((s) => s.id !== id);
    localStorage.setItem("regnav:sessions", JSON.stringify(updated));
    setSelected(null);
    setSessions(updated);
  };

  const clearAll = () => {
    localStorage.removeItem("regnav:sessions");
    setSessions([]);
    setSelected(null);
  };

  const exportAll = () => {
    const blob = new Blob([JSON.stringify(sessions, null, 2)], { type: "application/json" });
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

              {sessions.length === 0 ? (
                <div className="text-sm text-muted-foreground">No sessions found.</div>
              ) : (
                <div className="divide-y">
                  {sessions.map((s) => (
                    <div key={s.id} className="py-3 flex items-start justify-between">
                      <div className="cursor-pointer" onClick={() => setSelected(s)}>
                        <div className="font-semibold">{s.title}</div>
                        <div className="text-xs text-muted-foreground">{s.snippet}</div>
                      </div>
                      <div className="flex flex-col items-end gap-2">
                        <div className="text-xs text-muted-foreground">{new Date(s.time).toLocaleString()}</div>
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
                      <div className="text-xs text-muted-foreground">{new Date(selected.time).toLocaleString()}</div>
                    </div>
                    <div className="flex gap-2">
                      <Button onClick={() => openInChat(selected.id)}>Open in Chat</Button>
                      <Button variant="outline" onClick={() => remove(selected.id)}>Delete</Button>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {selected.messages.map((m: any) => (
                      <div key={m.id}>
                        <MessageBubble role={m.role}>{m.content}</MessageBubble>
                      </div>
                    ))}
                  </div>
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
