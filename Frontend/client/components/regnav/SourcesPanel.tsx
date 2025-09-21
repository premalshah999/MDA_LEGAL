import { ChatCitation } from "@shared/api";
import { SourceCard } from "./SourceCard";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

export function SourcesPanel({ latestHits }: { latestHits: ChatCitation[] }) {
  // Filter out empty/invalid citations and deduplicate by id/url/label
  const hits: ChatCitation[] = (() => {
    const input = Array.isArray(latestHits) ? latestHits : [];
    const filtered = input.filter((h) => {
      if (!h) return false;
      const hasLabel = typeof h.label === "string" && h.label.trim().length > 0;
      const hasContent = (typeof h.url === "string" && h.url.trim().length > 0) || (typeof h.snippet === "string" && h.snippet.trim().length > 0);
      return hasLabel && hasContent;
    });
    const seen = new Set<string>();
    const uniq: ChatCitation[] = [];
    for (const h of filtered) {
      const key = String(h.id ?? h.url ?? `${h.label}|${h.docId ?? ""}`);
      if (!seen.has(key)) {
        seen.add(key);
        uniq.push(h);
      }
    }
    return uniq;
  })();

  return (
    <div className="w-full">
      <div className="mt-0">
        <Tabs defaultValue="citations" className="w-full">
          <TabsList className="grid w-full grid-cols-1">
            <TabsTrigger value="citations">Citations</TabsTrigger>
          </TabsList>
          <TabsContent value="citations" className="mt-3">
            <div className="max-h-[50vh] overflow-y-auto space-y-3 pr-1">
              {hits.length === 0 ? (
                <p className="text-sm text-muted-foreground">No citations yet. Ask a question to see sources.</p>
              ) : (
                hits.map((h) => <SourceCard key={h.id} hit={h} />)
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
