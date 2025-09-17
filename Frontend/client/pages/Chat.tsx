import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { SiteHeader } from "@/components/layout/SiteHeader";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { ChatPanel } from "@/components/regnav/ChatPanel";
import { ChatHistory } from "@/components/regnav/ChatHistory";
import { SearchFilters } from "@shared/api";

export default function Chat() {
  const [selected, setSelected] = useState<any | null>(null);
  const [filters] = useState<SearchFilters>({ jurisdiction: "Maryland" });
  const location = useLocation();

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const sessionId = params.get("session");
    if (sessionId) {
      try {
        const sessions = JSON.parse(localStorage.getItem("regnav:sessions") || "[]");
        const s = sessions.find((ss: any) => ss.id === sessionId);
        if (s) setSelected(s);
      } catch {
        setSelected(null);
      }
    }
  }, [location.search]);

  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6 lg:px-8">
        <section className="grid grid-cols-1 gap-6 md:grid-cols-12">
          <aside className="md:col-span-4">
            <div className="rounded-2xl border bg-card p-4">
              <ChatHistory onSelect={(s) => setSelected(s)} />
            </div>
          </aside>

          <div className="md:col-span-8">
            <div className="rounded-2xl border bg-card p-4">
              <ChatPanel key={selected?.id || "new"} filters={filters} initialMessages={selected?.messages} />
            </div>
          </div>
        </section>
      </main>

      <SiteFooter />
    </div>
  );
}
