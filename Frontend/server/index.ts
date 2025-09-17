import "dotenv/config";
import express from "express";
import cors from "cors";
import { handleDemo } from "./routes/demo";
import { searchRoute } from "./routes/search";
import { addDocumentRoute, listDocumentsRoute } from "./routes/documents";

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

  return app;
}
