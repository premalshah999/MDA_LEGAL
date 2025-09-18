import { AddDocumentRequest, RegDocument, RegDocumentMeta } from "@shared/api";
import { seedCorpus } from "../data/corpus";

let documents: RegDocument[] = [...seedCorpus];

export function listDocuments(): RegDocumentMeta[] {
  return documents.map(({ text, ...meta }) => meta);
}

export function addDocument(doc: AddDocumentRequest): RegDocumentMeta {
  const exists = documents.find((d) => d.id === doc.id);
  if (exists) {
    documents = documents.map((d) => (d.id === doc.id ? { ...doc } : d));
  } else {
    documents.push({ ...doc });
  }
  const { text, ...meta } = doc;
  return meta;
}
