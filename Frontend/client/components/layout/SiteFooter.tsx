export function SiteFooter() {
  return (
    <footer className="border-t bg-card/30">
      <div className="mx-auto max-w-7xl px-4 py-6 text-xs text-muted-foreground sm:px-6 lg:px-8">
        <div className="flex flex-col items-center justify-between gap-3 sm:flex-row">
          <p>
            Built for rapid, reproducible regulatory research. Maryland-focused sample corpus. No external services required.
          </p>
          <p className="text-[10px]">
            Not legal advice. Verify against primary sources.
          </p>
        </div>
      </div>
    </footer>
  );
}
