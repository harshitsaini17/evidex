"use client";

import { useState, useCallback } from "react";
import clsx from "clsx";
import { ExplainResponse } from "@/lib/types";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface AnswerCardProps {
  response: ExplainResponse;
  onCitationClick?: (paragraphId: string) => void;
  className?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Confidence Badge Component
// ─────────────────────────────────────────────────────────────────────────────

interface ConfidenceBadgeProps {
  confidence: "high" | "low";
}

function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  const isHigh = confidence === "high";

  return (
    <span
      className={clsx(
        "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border",
        isHigh
          ? "bg-green-100 text-green-800 border-green-300"
          : "bg-amber-100 text-amber-800 border-amber-300"
      )}
      aria-label={`Confidence level: ${confidence}`}
    >
      <span
        className={clsx(
          "w-1.5 h-1.5 rounded-full mr-1.5",
          isHigh ? "bg-green-500" : "bg-amber-500"
        )}
        aria-hidden="true"
      />
      {confidence}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Citation Pill Component
// ─────────────────────────────────────────────────────────────────────────────

interface CitationPillProps {
  citationId: string;
  onClick?: (citationId: string) => void;
}

function CitationPill({ citationId, onClick }: CitationPillProps) {
  // Truncate long IDs for display
  const displayId =
    citationId.length > 16 ? `${citationId.slice(0, 16)}…` : citationId;

  return (
    <button
      type="button"
      onClick={() => onClick?.(citationId)}
      data-qa-id={citationId}
      className={clsx(
        "inline-flex items-center px-2.5 py-1 rounded-full text-xs font-mono",
        "bg-blue-50 text-blue-700 border border-blue-200",
        "hover:bg-blue-100 hover:border-blue-300",
        "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1",
        "transition-colors cursor-pointer"
      )}
      aria-label={`View citation: ${citationId}`}
      title={citationId}
    >
      <svg
        className="w-3 h-3 mr-1 opacity-60"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
        />
      </svg>
      {displayId}
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Collapsible Debug Panel Component
// ─────────────────────────────────────────────────────────────────────────────

interface DebugPanelProps {
  debug: NonNullable<ExplainResponse["debug"]>;
}

function DebugPanel({ debug }: DebugPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const toggleExpanded = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  // Guard: only show IDs and reasons, never raw prompts
  // Truncate any suspiciously long content that might be a prompt
  const sanitizeReason = (reason: string | undefined): string | null => {
    if (!reason) return null;
    // If it looks like a raw prompt (very long or contains certain patterns), truncate
    if (reason.length > 500) {
      return reason.slice(0, 500) + "… [truncated]";
    }
    return reason;
  };

  const plannerReason = sanitizeReason(debug.planner_reason);
  const verifierReason = sanitizeReason(debug.verifier_reason);
  const hasEvidenceLinks =
    debug.evidence_links && debug.evidence_links.length > 0;

  // Don't render if no debug info
  if (!plannerReason && !verifierReason && !hasEvidenceLinks) {
    return null;
  }

  return (
    <div className="border-t border-gray-200 pt-3">
      <button
        type="button"
        onClick={toggleExpanded}
        className={clsx(
          "flex items-center gap-2 w-full text-left",
          "text-sm font-medium text-gray-600 hover:text-gray-800",
          "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 rounded"
        )}
        aria-expanded={isExpanded}
        aria-controls="debug-content"
      >
        <svg
          className={clsx(
            "w-4 h-4 transition-transform",
            isExpanded && "rotate-90"
          )}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 5l7 7-7 7"
          />
        </svg>
        <span>Debug Info</span>
        <span className="text-xs text-gray-400 font-normal">(for developers)</span>
      </button>

      {isExpanded && (
        <div
          id="debug-content"
          className="mt-3 pl-6 space-y-3 text-sm"
          role="region"
          aria-label="Debug information"
        >
          {/* Planner Reason */}
          {plannerReason && (
            <div>
              <dt className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Planner Reason
              </dt>
              <dd className="mt-1 text-gray-700 bg-gray-50 p-2 rounded border border-gray-100">
                {plannerReason}
              </dd>
            </div>
          )}

          {/* Verifier Reason */}
          {verifierReason && (
            <div>
              <dt className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Verifier Reason
              </dt>
              <dd className="mt-1 text-gray-700 bg-gray-50 p-2 rounded border border-gray-100">
                {verifierReason}
              </dd>
            </div>
          )}

          {/* Evidence Links (IDs only) */}
          {hasEvidenceLinks && (
            <div>
              <dt className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Evidence Links
              </dt>
              <dd className="mt-1">
                <ul className="space-y-1">
                  {debug.evidence_links!.map((link, idx) => (
                    <li
                      key={idx}
                      className="text-xs font-mono text-gray-600 bg-gray-50 p-1.5 rounded border border-gray-100"
                    >
                      <span className="text-gray-400">Sources:</span>{" "}
                      {link.source_ids.join(", ")}
                    </li>
                  ))}
                </ul>
              </dd>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// AnswerCard Component
// ─────────────────────────────────────────────────────────────────────────────

export default function AnswerCard({
  response,
  onCitationClick,
  className,
}: AnswerCardProps) {
  const { answer, citations, confidence, debug } = response;

  return (
    <article
      className={clsx(
        "bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden",
        className
      )}
      aria-label="Answer to your question"
    >
      {/* Header with confidence badge */}
      <header className="flex items-center justify-between px-4 py-3 bg-gray-50 border-b border-gray-200">
        <h3 className="text-sm font-semibold text-gray-700">Answer</h3>
        <ConfidenceBadge confidence={confidence} />
      </header>

      {/* Answer content */}
      <div className="p-4 space-y-4">
        {/* Answer text with preserved newlines */}
        <div className="prose prose-sm max-w-none">
          <p className="text-gray-800 whitespace-pre-wrap leading-relaxed">
            {answer}
          </p>
        </div>

        {/* Citations */}
        {citations.length > 0 && (
          <section aria-labelledby="citations-heading">
            <h4
              id="citations-heading"
              className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2"
            >
              Citations ({citations.length})
            </h4>
            <ul className="flex flex-wrap gap-2" role="list">
              {citations.map((citationId) => (
                <li key={citationId}>
                  <CitationPill
                    citationId={citationId}
                    onClick={onCitationClick}
                  />
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* Debug Panel (collapsible) */}
        {debug && <DebugPanel debug={debug} />}
      </div>
    </article>
  );
}
