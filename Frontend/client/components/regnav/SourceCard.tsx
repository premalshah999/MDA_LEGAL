import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ExternalLink } from "lucide-react";
import { ChatCitation } from "@shared/api";

export function SourceCard({ hit }: { hit: ChatCitation }) {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="space-y-1 pb-2">
        <CardTitle className="text-sm font-semibold leading-snug">
          {hit.docTitle ?? hit.label}
        </CardTitle>
        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
          {hit.comarDisplay && <Badge variant="secondary">{hit.comarDisplay}</Badge>}
          {hit.pages && <Badge variant="outline">Pages {hit.pages}</Badge>}
        </div>
      </CardHeader>
      <CardContent>
        {hit.snippet && <p className="line-clamp-4 text-xs leading-relaxed">{hit.snippet}</p>}
        {hit.url && (
          <a
            className="mt-2 inline-flex items-center gap-1 text-xs text-primary hover:underline"
            href={hit.url}
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
