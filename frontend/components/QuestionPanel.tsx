"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import clsx from "clsx";
import { explainDocument } from "@/lib/api";
import { ExplainRequest, ExplainResponse } from "@/lib/types";
import AnswerCard from "./AnswerCard";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface QuestionPanelProps {
  documentId: string;
  onShowAnswer?: (response: ExplainResponse) => void;
  onCitationClick?: (citationId: string) => void;
  className?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Toast Component
// ─────────────────────────────────────────────────────────────────────────────

interface ToastProps {
  message: string;
  type: "info" | "error" | "warning";
  onDismiss: () => void;
}

function Toast({ message, type, onDismiss }: ToastProps) {
  useEffect(() => {
    const timer = setTimeout(onDismiss, 4000);
    return () => clearTimeout(timer);
  }, [onDismiss]);

  const bgColor = {
    info: "bg-blue-600",
    error: "bg-red-600",
    warning: "bg-yellow-500",
  }[type];

  return (
    <div
      role="alert"
      aria-live="polite"
      className={clsx(
        "fixed bottom-4 right-4 px-4 py-3 rounded-lg shadow-lg text-white z-50",
        "animate-slide-in-up",
        bgColor
      )}
    >
      <div className="flex items-center gap-3">
        <span>{message}</span>
        <button
          onClick={onDismiss}
          className="text-white/80 hover:text-white"
          aria-label="Dismiss notification"
        >
          ✕
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-resize Textarea Hook
// ─────────────────────────────────────────────────────────────────────────────

function useAutoResize(value: string) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [value]);

  return textareaRef;
}

// ─────────────────────────────────────────────────────────────────────────────
// QuestionPanel Component
// ─────────────────────────────────────────────────────────────────────────────

const MAX_QUESTION_LENGTH = 2000;

export default function QuestionPanel({
  documentId,
  onShowAnswer,
  onCitationClick,
  className,
}: QuestionPanelProps) {
  // Form state
  const [question, setQuestion] = useState("");
  const [paragraphIdsInput, setParagraphIdsInput] = useState("");
  const [includeDebug, setIncludeDebug] = useState(false);

  // Request state
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<ExplainResponse | null>(null);

  // Toast state
  const [toast, setToast] = useState<{
    message: string;
    type: "info" | "error" | "warning";
  } | null>(null);

  // Auto-resize textarea
  const textareaRef = useAutoResize(question);

  // Validation
  const questionTrimmed = question.trim();
  const isValid = questionTrimmed.length > 0 && questionTrimmed.length <= MAX_QUESTION_LENGTH;
  const charCount = questionTrimmed.length;
  const charCountWarning = charCount > MAX_QUESTION_LENGTH * 0.9;

  // Parse paragraph IDs from comma-separated input
  const parseParagraphIds = useCallback((input: string): string[] => {
    if (!input.trim()) return [];
    return input
      .split(",")
      .map((id) => id.trim())
      .filter((id) => id.length > 0);
  }, []);

  // Handle form submission
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      if (!isValid || isLoading) return;

      setError(null);
      setIsLoading(true);

      try {
        // Build request
        const paragraphIds = parseParagraphIds(paragraphIdsInput);
        const payload: ExplainRequest = {
          question: questionTrimmed,
          ...(paragraphIds.length > 0 && { paragraph_ids: paragraphIds }),
          ...(includeDebug && { include_debug: true }),
        };

        // Call API
        const result = await explainDocument(documentId, payload);

        // Check for "not defined" type responses
        const notDefinedPhrases = [
          "not defined in the paper",
          "not mentioned in the document",
          "cannot find information",
          "no information available",
        ];
        const isNotDefined = notDefinedPhrases.some((phrase) =>
          result.answer.toLowerCase().includes(phrase)
        );

        if (isNotDefined) {
          setToast({
            message: "This information is not defined in the paper",
            type: "warning",
          });
        }

        // Store response and notify parent
        setResponse(result);
        onShowAnswer?.(result);
      } catch (err) {
        const errorMessage =
          err instanceof Error
            ? err.message
            : "Failed to get answer. Please try again.";
        setError(errorMessage);
        setToast({ message: errorMessage, type: "error" });
      } finally {
        setIsLoading(false);
      }
    },
    [
      isValid,
      isLoading,
      questionTrimmed,
      paragraphIdsInput,
      includeDebug,
      documentId,
      parseParagraphIds,
      onShowAnswer,
    ]
  );

  // Dismiss toast
  const dismissToast = useCallback(() => setToast(null), []);

  // Clear response when question changes significantly
  const handleQuestionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    if (newValue.length <= MAX_QUESTION_LENGTH + 100) {
      setQuestion(newValue);
    }
  };

  return (
    <div className={clsx("space-y-4", className)}>
      <form onSubmit={handleSubmit} className="space-y-3" aria-label="Ask a question about this document">
        {/* Question Textarea */}
        <div>
          <label
            htmlFor="question-input"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Your Question
          </label>
          <textarea
            ref={textareaRef}
            id="question-input"
            value={question}
            onChange={handleQuestionChange}
            placeholder="Ask a question about this document..."
            maxLength={MAX_QUESTION_LENGTH + 100}
            disabled={isLoading}
            aria-describedby="question-char-count"
            aria-invalid={charCount > MAX_QUESTION_LENGTH}
            className={clsx(
              "w-full px-3 py-2 border rounded-lg resize-none",
              "text-sm leading-relaxed",
              "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500",
              "transition-colors",
              isLoading && "opacity-50 cursor-not-allowed bg-gray-50",
              charCount > MAX_QUESTION_LENGTH && "border-red-500"
            )}
            style={{ minHeight: "80px" }}
          />
          <div
            id="question-char-count"
            className={clsx(
              "text-xs text-right mt-1",
              charCount > MAX_QUESTION_LENGTH
                ? "text-red-600"
                : charCountWarning
                ? "text-yellow-600"
                : "text-gray-400"
            )}
          >
            {charCount}/{MAX_QUESTION_LENGTH}
          </div>
        </div>

        {/* Paragraph IDs Input (Optional) */}
        <div>
          <label
            htmlFor="paragraph-ids-input"
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Paragraph IDs{" "}
            <span className="font-normal text-gray-500">(optional, comma-separated)</span>
          </label>
          <input
            type="text"
            id="paragraph-ids-input"
            value={paragraphIdsInput}
            onChange={(e) => setParagraphIdsInput(e.target.value)}
            placeholder="e.g., p1, p2, p3"
            disabled={isLoading}
            className={clsx(
              "w-full px-3 py-2 border rounded-lg text-sm",
              "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500",
              "transition-colors",
              isLoading && "opacity-50 cursor-not-allowed bg-gray-50"
            )}
          />
        </div>

        {/* Include Debug Checkbox */}
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="include-debug-checkbox"
            checked={includeDebug}
            onChange={(e) => setIncludeDebug(e.target.checked)}
            disabled={isLoading}
            className={clsx(
              "h-4 w-4 rounded border-gray-300 text-blue-600",
              "focus:ring-2 focus:ring-blue-500 focus:ring-offset-1",
              isLoading && "opacity-50 cursor-not-allowed"
            )}
          />
          <label
            htmlFor="include-debug-checkbox"
            className="text-sm text-gray-700 select-none cursor-pointer"
          >
            Include debug info
          </label>
        </div>

        {/* Error Message */}
        {error && (
          <div
            role="alert"
            className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700"
          >
            {error}
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!isValid || isLoading}
          aria-busy={isLoading}
          className={clsx(
            "w-full px-4 py-2.5 rounded-lg font-medium text-sm",
            "flex items-center justify-center gap-2",
            "transition-colors",
            isValid && !isLoading
              ? "bg-blue-600 text-white hover:bg-blue-700"
              : "bg-gray-300 text-gray-500 cursor-not-allowed"
          )}
        >
          {isLoading ? (
            <>
              <svg
                className="animate-spin h-4 w-4"
                xmlns="http://www.w3.org/2000/svg"
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
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              <span>Processing...</span>
            </>
          ) : (
            "Ask"
          )}
        </button>
      </form>

      {/* Answer Card */}
      {response && (
        <AnswerCard
          response={response}
          onCitationClick={onCitationClick}
          className="mt-4"
        />
      )}

      {/* Toast Notification */}
      {toast && (
        <Toast message={toast.message} type={toast.type} onDismiss={dismissToast} />
      )}
    </div>
  );
}
