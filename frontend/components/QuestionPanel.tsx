"use client";

import { useState } from "react";
import clsx from "clsx";

export interface QuestionPanelProps {
  documentId: string;
  onAsk?: (question: string) => void;
  isLoading?: boolean;
  className?: string;
}

export default function QuestionPanel({
  documentId,
  onAsk,
  isLoading = false,
  className,
}: QuestionPanelProps) {
  const [question, setQuestion] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (question.trim() && !isLoading) {
      onAsk?.(question.trim());
    }
  }

  // TODO: Implement question panel logic
  return (
    <div className={clsx("p-4 border rounded-lg bg-white", className)}>
      <h3 className="text-lg font-semibold mb-3">Ask a Question</h3>
      <form onSubmit={handleSubmit} className="space-y-3">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about this document..."
          rows={3}
          disabled={isLoading}
          className={clsx(
            "w-full px-3 py-2 border rounded resize-none",
            "focus:outline-none focus:ring-2 focus:ring-blue-500",
            isLoading && "opacity-50 cursor-not-allowed"
          )}
        />
        <button
          type="submit"
          disabled={!question.trim() || isLoading}
          className={clsx(
            "w-full px-4 py-2 bg-blue-600 text-white rounded font-medium",
            "hover:bg-blue-700 transition-colors",
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
        >
          {isLoading ? "Processing..." : "Ask"}
        </button>
      </form>
      <input type="hidden" name="documentId" value={documentId} />
    </div>
  );
}
