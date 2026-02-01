"use client";

import clsx from "clsx";
import { Paragraph } from "@/lib/types";

export interface ParagraphItemProps {
  /** The paragraph ID */
  paragraphId: string;
  /** Cached paragraph data (if loaded) */
  paragraph?: Paragraph;
  /** Whether this paragraph is currently selected */
  isSelected?: boolean;
  /** Whether this paragraph is currently loading */
  isLoading?: boolean;
  /** Callback when paragraph is selected */
  onSelect: () => void;
  /** Additional CSS classes */
  className?: string;
}

export default function ParagraphItem({
  paragraphId,
  paragraph,
  isSelected,
  isLoading,
  onSelect,
  className,
}: ParagraphItemProps) {
  // Extract a short excerpt from text (first ~80 chars)
  const excerpt = paragraph?.text
    ? paragraph.text.length > 80
      ? paragraph.text.slice(0, 80).trim() + "..."
      : paragraph.text
    : null;

  return (
    <div
      className={clsx(
        "px-3 py-2 cursor-pointer transition-colors",
        isSelected
          ? "bg-blue-50 border-l-2 border-blue-500"
          : "hover:bg-gray-50 border-l-2 border-transparent",
        className
      )}
    >
      <div className="flex items-start justify-between gap-2">
        {/* Paragraph ID and excerpt */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span
              className={clsx(
                "text-xs font-mono",
                isSelected ? "text-blue-700" : "text-gray-500"
              )}
              title={paragraphId}
            >
              {paragraphId.length > 20
                ? paragraphId.slice(0, 8) + "..." + paragraphId.slice(-8)
                : paragraphId}
            </span>
            
            {/* Page badge */}
            {paragraph?.page && (
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs bg-gray-100 text-gray-600">
                p.{paragraph.page}
              </span>
            )}
          </div>
          
          {/* Excerpt (if available) */}
          {excerpt && (
            <p className="mt-1 text-xs text-gray-600 line-clamp-2">{excerpt}</p>
          )}
        </div>

        {/* View button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
          disabled={isLoading}
          className={clsx(
            "shrink-0 px-2 py-1 text-xs font-medium rounded",
            "transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500",
            isLoading
              ? "bg-gray-100 text-gray-400 cursor-not-allowed"
              : isSelected
                ? "bg-blue-600 text-white hover:bg-blue-700"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          )}
          aria-label={`View paragraph ${paragraphId}`}
        >
          {isLoading ? (
            <span className="flex items-center gap-1">
              <svg
                className="animate-spin h-3 w-3"
                fill="none"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              ...
            </span>
          ) : paragraph ? (
            "View"
          ) : (
            "Load"
          )}
        </button>
      </div>

      {/* Click area for the whole item */}
      <button
        onClick={onSelect}
        className="absolute inset-0 w-full h-full opacity-0"
        aria-hidden="true"
        tabIndex={-1}
      />
    </div>
  );
}
