import { cn } from "@/lib/utils";
import { Link } from "react-router-dom";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";

export function SiteHeader({ className }: { className?: string }) {
  return (
    <header className={cn("sticky top-0 z-40 w-full border-b bg-background/80 backdrop-blur", className)}>
      <div className="mx-auto flex h-20 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link to="/" className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-md bg-transparent text-primary overflow-hidden">
            <img src="https://cdn.builder.io/api/v1/image/assets%2F4d53304e2490480db08b5726aee537a0%2Fd57d7ae53e1b4f1d9b09c10bf7f74e99?format=webp&width=800" alt="RegNav logo" className="h-10 w-10 object-contain" />
          </div>
          <div className="flex flex-col leading-tight">
            <h1 className="text-lg font-bold tracking-tight">RegNav AI</h1>
          </div>
        </Link>
        <div className="hidden items-center gap-2 sm:flex">
          <Button asChild variant="ghost" className="gap-2">
            <a href="https://dsd.maryland.gov/Pages/COMARSearch.aspx" target="_blank" rel="noreferrer">Corpus</a>
          </Button>
          <Button asChild className="gap-2" variant="default">
            <a href="https://6dbf694181fa45e2bfdef44b20945e15-9eda4089f2e54c418ee1a6e13.fly.dev/#try" target="_blank" rel="noreferrer"><Search className="h-4 w-4" /> Try it</a>
          </Button>
          <Button asChild variant="link" className="gap-2">
            <Link to="/admin">Admin</Link>
          </Button>
          <Button asChild variant="link" className="gap-2">
            <Link to="/login">Log in</Link>
          </Button>
          <Button asChild className="gap-2">
            <Link to="/signup">Sign up</Link>
          </Button>
        </div>
      </div>
    </header>
  );
}
