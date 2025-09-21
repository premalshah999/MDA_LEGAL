import { RequestHandler } from "express";
import { ChatAskRequest } from "@shared/api";
import { askBackend } from "../services/backend";

export const searchRoute: RequestHandler = async (req, res) => {
  const userId = req.header("x-user-id");
  if (!userId) {
    return res.status(400).json({ error: "Missing X-User-Id header" });
  }

  const body = req.body as ChatAskRequest;
  if (!body || typeof body.question !== "string" || body.question.trim().length === 0) {
    return res.status(400).json({ error: "Missing 'question'" });
  }

  try {
    const response = await askBackend(userId, body);
    res.status(200).json(response);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Backend error";
    res.status(502).json({ error: message });
  }
};
