import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Jurisdiction, SearchFilters } from "@shared/api";

export function FiltersPanel({
  value,
  onChange,
  onReset,
}: {
  value: SearchFilters;
  onChange: (v: SearchFilters) => void;
  onReset: () => void;
}) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label>Jurisdiction</Label>
        <Select
          value={value.jurisdiction ?? "any"}
          onValueChange={(v) => onChange({ ...value, jurisdiction: v === "any" ? undefined : (v as Jurisdiction) })}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Any" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="any">Any</SelectItem>
            <SelectItem value="Maryland">Maryland</SelectItem>
            <SelectItem value="Federal">Federal</SelectItem>
            <SelectItem value="Other">Other</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label>Agency</Label>
        <Input
          placeholder="e.g., Maryland Department of Agriculture"
          value={value.agency ?? ""}
          onChange={(e) => onChange({ ...value, agency: e.target.value || undefined })}
        />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-2">
          <Label>Year From</Label>
          <Input
            type="number"
            inputMode="numeric"
            min={1900}
            max={2100}
            value={value.yearFrom ?? ""}
            onChange={(e) => onChange({ ...value, yearFrom: e.target.value ? Number(e.target.value) : undefined })}
          />
        </div>
        <div className="space-y-2">
          <Label>Year To</Label>
          <Input
            type="number"
            inputMode="numeric"
            min={1900}
            max={2100}
            value={value.yearTo ?? ""}
            onChange={(e) => onChange({ ...value, yearTo: e.target.value ? Number(e.target.value) : undefined })}
          />
        </div>
      </div>

      <div className="pt-1">
        <Button variant="outline" size="sm" onClick={onReset} className="w-full">
          Reset filters
        </Button>
      </div>
    </div>
  );
}
