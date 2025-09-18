import { ChatCitation } from "@shared/api";
import { SourceCard } from "./SourceCard";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

export function SourcesPanel({ latestHits }: { latestHits: ChatCitation[] }) {
  return (
    <div className="w-full">
      <div className="mt-0">
        <Tabs defaultValue="citations" className="w-full">
          <TabsList className="grid w-full grid-cols-1">
            <TabsTrigger value="citations">Citations</TabsTrigger>
          </TabsList>
          <TabsContent value="citations" className="mt-3 space-y-3">
            {latestHits.length === 0 ? (
              <p className="text-sm text-muted-foreground">Run a query to see top-matching sources.</p>
            ) : (
              latestHits.map((h) => <SourceCard key={h.id} hit={h} />)
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
