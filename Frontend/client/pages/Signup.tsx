import { Link } from "react-router-dom";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { SiteHeader } from "@/components/layout/SiteHeader";
import { SiteFooter } from "@/components/layout/SiteFooter";

export default function Signup() {
  return (
    <div className="min-h-screen bg-background">
      <SiteHeader />

      <main className="mx-auto max-w-md p-6">
        <div className="rounded-2xl border bg-card p-6 shadow-md">
          <h2 className="text-2xl font-semibold">Create account</h2>
          <p className="mt-2 text-sm text-muted-foreground">Create an account to save searches and share citations.</p>

          <div className="mt-6 space-y-4">
            <div>
              <label className="mb-1 block text-sm font-medium">Full name</label>
              <Input placeholder="Jane Doe" />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium">Email</label>
              <Input type="email" placeholder="you@agency.gov" />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium">Password</label>
              <Input type="password" placeholder="Choose a strong password" />
            </div>
            <div className="flex items-center justify-between">
              <Button>Sign up</Button>
              <Link to="/login" className="text-sm text-muted-foreground hover:underline">Already have an account?</Link>
            </div>
          </div>
        </div>
      </main>

      <SiteFooter />
    </div>
  );
}
