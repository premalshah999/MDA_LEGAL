const USER_KEY = "regnav:userId";

export function ensureUserId(): string {
  if (typeof window === "undefined") {
    return "";
  }
  const existing = window.localStorage.getItem(USER_KEY);
  if (existing) {
    return existing;
  }
  const id = crypto.randomUUID();
  window.localStorage.setItem(USER_KEY, id);
  return id;
}
