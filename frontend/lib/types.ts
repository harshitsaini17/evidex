/**
 * Shared TypeScript types for Evidex frontend
 */

export interface DocumentInfo {
  document_id: string;
  title: string;
  status: string;
}

export interface SectionSummary {
  section_title: string;
  paragraph_ids: string[];
}

export interface Paragraph {
  paragraph_id: string;
  text?: string;
  page?: number;
  equation_refs?: string[];
}

export interface ExplainRequest {
  question: string;
  paragraph_ids?: string[];
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
