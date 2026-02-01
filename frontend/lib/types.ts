/**
 * Shared TypeScript types for Evidex frontend
 */

export interface DocumentInfo {
  document_id: string;
  title: string;
  status: string;
}

export interface SectionSummary {
  title: string;
  paragraph_ids: string[];
}

export interface SectionsResponse {
  document_id: string;
  sections: SectionSummary[];
}

export interface Paragraph {
  paragraph_id: string;
  text?: string;
  section_title?: string;
  page?: number;
  equation_refs?: string[];
}

export interface ExplainRequest {
  question: string;
  paragraph_ids?: string[];
  context_text?: string; // Selected text as context
  include_debug?: boolean;
}

export interface ExplainResponseDebug {
  planner_reason?: string;
  verifier_reason?: string;
  evidence_links?: Array<{ source_ids: string[] }>;
}

export interface ExplainResponse {
  answer: string;
  citations: string[];
  confidence: "high" | "low";
  debug?: ExplainResponseDebug;
}

// ─────────────────────────────────────────────────────────────────────────────
// Selection Types for PDF Context
// ─────────────────────────────────────────────────────────────────────────────

export interface TextSelection {
  type: 'text';
  text: string;
  page: number;
}

export interface RectSelection {
  type: 'rect';
  page: number;
  bbox: { x: number; y: number; width: number; height: number };
  extractedText?: string; // Text extracted from the region
}

export type ContextSelection = TextSelection | RectSelection;

