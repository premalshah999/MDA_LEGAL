import { RequestHandler } from "express";
import {
  addMemory,
  createSession,
  getSession,
  listMemories,
  listSessions,
  removeMemory,
  removeSession,
  renameSession,
} from "../services/backend";

function requireUserId(req: Parameters<RequestHandler>[0], res: Parameters<RequestHandler>[1]): string | undefined {
  const userId = req.header("x-user-id");
  if (!userId) {
    res.status(400).json({ error: "Missing X-User-Id header" });
    return undefined;
  }
  return userId;
}

export const listSessionsRoute: RequestHandler = async (req, res) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  try {
    const sessions = await listSessions(userId);
    res.status(200).json({ sessions });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend error";
    res.status(502).json({ error: message });
  }
};

export const getSessionRoute: RequestHandler = async (req, res) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  const { sessionId } = req.params;
  if (!sessionId) {
    return res.status(400).json({ error: "Missing sessionId parameter" });
  }
  try {
    const session = await getSession(userId, sessionId);
    res.status(200).json({ session });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend error";
    res.status(message.includes("not found") ? 404 : 502).json({ error: message });
  }
};

export const createSessionRoute: RequestHandler = async (req, res) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  const title = typeof req.body?.title === "string" ? req.body.title : undefined;
  try {
    const session = await createSession(userId, title);
    res.status(201).json({ session });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend error";
    res.status(502).json({ error: message });
  }
};

export const renameSessionRoute: RequestHandler = async (req, res) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  const { sessionId } = req.params;
  const title = typeof req.body?.title === "string" ? req.body.title : "";
  if (!sessionId) {
    return res.status(400).json({ error: "Missing sessionId parameter" });
  }
  if (!title.trim()) {
    return res.status(400).json({ error: "Title must not be empty" });
  }
  try {
    const session = await renameSession(userId, sessionId, title);
    res.status(200).json({ session });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend error";
    res.status(message.includes("not found") ? 404 : 502).json({ error: message });
  }
};

export const deleteSessionRoute: RequestHandler = async (req, res) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  const { sessionId } = req.params;
  if (!sessionId) {
    return res.status(400).json({ error: "Missing sessionId parameter" });
  }
  try {
    await removeSession(userId, sessionId);
    res.status(204).send();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend error";
    res.status(message.includes("not found") ? 404 : 502).json({ error: message });
  }
};

export const listMemoriesRoute: RequestHandler = async (req, res) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  try {
    const memories = await listMemories(userId);
    res.status(200).json({ memories });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend error";
    res.status(502).json({ error: message });
  }
};

export const addMemoryRoute: RequestHandler = async (req, res) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  const content = typeof req.body?.content === "string" ? req.body.content : "";
  if (!content.trim()) {
    return res.status(400).json({ error: "Memory content must not be empty" });
  }
  try {
    const memory = await addMemory(userId, content);
    res.status(201).json({ memory });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend error";
    res.status(502).json({ error: message });
  }
};

export const deleteMemoryRoute: RequestHandler = async (req, res) => {
  const userId = requireUserId(req, res);
  if (!userId) return;
  const { memoryId } = req.params;
  if (!memoryId) {
    return res.status(400).json({ error: "Missing memoryId parameter" });
  }
  try {
    await removeMemory(userId, memoryId);
    res.status(204).send();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend error";
    res.status(message.includes("not found") ? 404 : 502).json({ error: message });
  }
};
