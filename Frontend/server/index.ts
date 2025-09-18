import "dotenv/config";
import express from "express";
import cors from "cors";
import { handleDemo } from "./routes/demo";
import { searchRoute } from "./routes/search";
import { addDocumentRoute, listDocumentsRoute } from "./routes/documents";
import {
  addMemoryRoute,
  createSessionRoute,
  deleteMemoryRoute,
  deleteSessionRoute,
  getSessionRoute,
  listMemoriesRoute,
  listSessionsRoute,
  renameSessionRoute,
} from "./routes/chat";

export function createServer() {
  const app = express();

  // Middleware
  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // Health and demo
  app.get("/api/ping", (_req, res) => {
    const ping = process.env.PING_MESSAGE ?? "ping";
    res.json({ message: ping });
  });

  app.get("/api/demo", handleDemo);

  // Regulatory chatbot endpoints
  app.post("/api/search", searchRoute);
  app.get("/api/documents", listDocumentsRoute);
  app.post("/api/documents", addDocumentRoute);
  app.get("/api/chat/sessions", listSessionsRoute);
  app.post("/api/chat/sessions", createSessionRoute);
  app.get("/api/chat/sessions/:sessionId", getSessionRoute);
  app.patch("/api/chat/sessions/:sessionId", renameSessionRoute);
  app.delete("/api/chat/sessions/:sessionId", deleteSessionRoute);
  app.get("/api/chat/memories", listMemoriesRoute);
  app.post("/api/chat/memories", addMemoryRoute);
  app.delete("/api/chat/memories/:memoryId", deleteMemoryRoute);

  return app;
}
