import { RequestHandler } from "express";
import { SearchRequest, SearchResponse } from "@shared/api";
import { handleSearch } from "../services/search";

export const searchRoute: RequestHandler = (req, res) => {
  const body = req.body as SearchRequest;
  if (!body || typeof body.query !== "string" || body.query.trim().length === 0) {
    return res.status(400).json({ error: "Missing 'query'" });
  }
  const response: SearchResponse = handleSearch(body);
  res.status(200).json(response);
};
