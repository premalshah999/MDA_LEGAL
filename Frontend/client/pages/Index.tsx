import { useState, useEffect } from "react";
import { SiteHeader } from "@/components/layout/SiteHeader";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { ChatPanel } from "@/components/regnav/ChatPanel";
import { SourcesPanel } from "@/components/regnav/SourcesPanel";
import { FiltersPanel } from "@/components/regnav/FiltersPanel";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import { SearchFilters, ChatCitation } from "@shared/api";
import { createDefaultSearchFilters } from "@/lib/searchFilters";
import { ensureUserId } from "@/lib/user";

export default function Index() {
  const [filters, setFilters] = useState<SearchFilters>(() => createDefaultSearchFilters());
  const [latestHits, setLatestHits] = useState<ChatCitation[]>([]);
  const [userId, setUserId] = useState<string>("");

  useEffect(() => {
    const handler = (e: Event) => {
      const ce = e as CustomEvent<ChatCitation[]>;
      setLatestHits(ce.detail || []);
    };
    window.addEventListener("regnav:citations", handler as EventListener);
    return () => window.removeEventListener("regnav:citations", handler as EventListener);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setUserId(ensureUserId());
  }, []);

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <SiteHeader />

      <main id="try" className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6 lg:px-8">
        <section className="mb-10 grid items-center gap-6 rounded-2xl border bg-card p-6 md:grid-cols-2">
          <div className="space-y-4">
            <div className="text-3xl font-extrabold tracking-tight sm:text-4xl">
              <p>
                <em style={{ color: "rgb(55, 89, 35)" }}>
                  Traceable, Reliable, and Grounded in COMAR
                </em>
              </p>
            </div>
            <p className="text-muted-foreground">
              Reliable access to Maryland regulations, powered by AI — fast,
              accurate, and traceable.
            </p>
            <div className="flex flex-wrap gap-2 text-sm text-muted-foreground">
              <div className="block font-normal">Deterministic synthesis</div>
              <div className="block">•</div>
              <div className="block"> <p>Citations</p> </div>
            </div>
            <div className="pt-2">
              <Button asChild variant="default">
                <a href="https://6dbf694181fa45e2bfdef44b20945e15-9eda4089f2e54c418ee1a6e13.fly.dev/#workspace" target="_blank" rel="noreferrer" className="inline-flex items-center gap-2">Start querying <ArrowRight className="h-4 w-4" /></a>
              </Button>
            </div>
          </div>
        </section>

        <section id="workspace" className="grid grid-cols-1 gap-6 md:grid-cols-12">
          <div className="md:col-span-8">
            <div className="rounded-2xl border bg-card p-4">
              <ChatPanel filters={filters} userId={userId} session={null} />
            </div>
          </div>
          <aside className="md:col-span-4">
            <div className="space-y-4">
              <div className="rounded-2xl border bg-card p-4">
                <FiltersPanel value={filters} onChange={setFilters} onReset={() => setFilters(createDefaultSearchFilters())} />
              </div>
              <div className="rounded-2xl border bg-card p-4">
                <SourcesPanel latestHits={latestHits} />
              </div>
            </div>
          </aside>
        </section>
      </main>

      {/* Chat moved to its own page */}
      <section id="chat-cta" className="bg-background">
        <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
          <div className="rounded-2xl border bg-card p-6">
            <h2 className="text-lg font-semibold mb-4">Chat interface moved</h2>
            <p className="text-sm text-muted-foreground mb-4">Open the full chat page to view history and start new chats.</p>
            <div>
              <a href="/chat" className="inline-flex items-center rounded-md bg-primary px-4 py-2 text-white">Open Chat</a>
            </div>
          </div>
        </div>
      </section>

      <SiteFooter />
    </div>
  );
}
