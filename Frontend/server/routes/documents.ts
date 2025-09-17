import { RequestHandler } from "express";
import { AddDocumentRequest, AddDocumentResponse, ListDocumentsResponse } from "@shared/api";
import { addDocument, listDocuments } from "../services/search";

export const listDocumentsRoute: RequestHandler = (_req, res) => {
  const docs = listDocuments();
  const response: ListDocumentsResponse = { documents: docs };
  res.status(200).json(response);
};

export const addDocumentRoute: RequestHandler = (req, res) => {
  const body = req.body as AddDocumentRequest;
  if (!body || !body.id || !body.title || !body.agency || !body.jurisdiction || !body.text || !body.year) {
    return res.status(400).json({ error: "id, title, agency, jurisdiction, year, text are required" });
  }
  const meta = addDocument(body);
  const response: AddDocumentResponse = { ok: true, document: meta };
  res.status(201).json(response);
};
