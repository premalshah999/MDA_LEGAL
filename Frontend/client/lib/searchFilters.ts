import { SearchFilters } from "@shared/api";

export const DEFAULT_SEARCH_FILTERS: SearchFilters = {
  jurisdiction: "Maryland",
};

export const createDefaultSearchFilters = (): SearchFilters => ({
  ...DEFAULT_SEARCH_FILTERS,
});
