"use client";

import clsx from "clsx";

export interface AnswerCardProps {
  question: string;
  answer: string;
  confidence: "high" | "low";
  citations?: string[];
  className?: string;
}

export default function AnswerCard({
  question,
  answer,
  confidence,
  citations = [],
  className,
}: AnswerCardProps) {
  // Determine confidence color
  const confidenceColor = confidence === "high" ? "text-green-600" : "text-yellow-600";

  // TODO: Implement full answer card logic
  return (
    <div className={clsx("p-4 border rounded-lg bg-white shadow-sm", className)}>
      {/* Question */}
      <div className="mb-3">
        <span className="text-xs text-gray-500 uppercase tracking-wide">Question</span>
        <p className="font-medium text-gray-900">{question}</p>
      </div>

      {/* Answer */}
      <div className="mb-3">
        <span className="text-xs text-gray-500 uppercase tracking-wide">Answer</span>
        <p className="text-gray-700">{answer}</p>
      </div>

      {/* Confidence */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs text-gray-500">Confidence:</span>
        <span className={clsx("text-sm font-medium", confidenceColor)}>{confidence}</span>
      </div>

      {/* Citations */}
      {citations.length > 0 && (
        <div>
          <span className="text-xs text-gray-500 uppercase tracking-wide">Citations</span>
          <ul className="mt-1 space-y-1">
            {citations.map((citation, index) => (
              <li key={index} className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                {citation}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
