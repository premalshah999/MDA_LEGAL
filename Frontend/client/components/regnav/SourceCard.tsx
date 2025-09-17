import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ExternalLink } from "lucide-react";
import { ScoredDoc } from "@shared/api";

export function SourceCard({ hit }: { hit: ScoredDoc }) {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="space-y-1 pb-2">
        <CardTitle className="text-sm font-semibold leading-snug">
          {hit.meta.title}
        </CardTitle>
        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
          <Badge variant="secondary">{hit.meta.agency}</Badge>
          <Badge variant="outline">{hit.meta.jurisdiction}</Badge>
          <Badge variant="outline">{hit.meta.year}</Badge>
          <span className="ml-auto text-[10px]">Score: {hit.score.toFixed(2)}</span>
        </div>
      </CardHeader>
      <CardContent>
        <p className="line-clamp-4 text-xs leading-relaxed">{hit.snippet.text}</p>
        {hit.meta.sourceUrl && (
          <a
            className="mt-2 inline-flex items-center gap-1 text-xs text-primary hover:underline"
            href={hit.meta.sourceUrl}
            target="_blank"
            rel="noreferrer"
          >
            View source <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </CardContent>
    </Card>
  );
}
