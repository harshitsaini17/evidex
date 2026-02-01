"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import clsx from "clsx";
import { explainDocument } from "@/lib/api";
import { ExplainRequest, ExplainResponse, ContextSelection } from "@/lib/types";
import AnswerCard from "./AnswerCard";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface QuestionPanelProps {
  documentId: string;
  /** Selected context from PDF viewer */
  selectedContext?: ContextSelection[];
  /** Callback to clear a specific selection */
  onClearSelection?: (index: number) => void;
  /** Callback to clear all selections */
  onClearAllSelections?: () => void;
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
// Context Badge Component
// ─────────────────────────────────────────────────────────────────────────────

interface ContextBadgeProps {
  selection: ContextSelection;
  index: number;
  onRemove: () => void;
}

function ContextBadge({ selection, index, onRemove }: ContextBadgeProps) {
  const text = selection.type === 'text'
    ? selection.text
    : selection.extractedText || 'Region selected';

  const truncatedText = text.length > 50 ? text.slice(0, 50) + '...' : text;

  return (
    <div
      className={clsx(
        "group flex items-start gap-2 p-2 rounded-lg border transition-colors",
        selection.type === 'text'
          ? "bg-blue-50 border-blue-200"
          : "bg-purple-50 border-purple-200"
      )}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5 mb-1">
          <span
            className={clsx(
              "text-xs font-medium px-1.5 py-0.5 rounded",
              selection.type === 'text'
                ? "bg-blue-100 text-blue-700"
                : "bg-purple-100 text-purple-700"
            )}
          >
            {selection.type === 'text' ? 'Text' : 'Region'}
          </span>
          <span className="text-xs text-gray-500">
            Page {selection.page}
          </span>
        </div>
        <p className="text-xs text-gray-700 line-clamp-2" title={text}>
          {truncatedText}
        </p>
      </div>
      <button
        onClick={onRemove}
        className={clsx(
          "shrink-0 w-5 h-5 flex items-center justify-center rounded-full",
          "text-gray-400 hover:text-white transition-colors",
          selection.type === 'text'
            ? "hover:bg-blue-500"
            : "hover:bg-purple-500"
        )}
        aria-label={`Remove context ${index + 1}`}
      >
        ×
      </button>
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
  selectedContext = [],
  onClearSelection,
  onClearAllSelections,
  onShowAnswer,
  onCitationClick,
  className,
}: QuestionPanelProps) {
  // Form state
  const [question, setQuestion] = useState("");
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

  // Build context text from selections
  const buildContextText = useCallback((): string => {
    if (selectedContext.length === 0) return '';

    const contextParts = selectedContext.map((sel, idx) => {
      const text = sel.type === 'text' ? sel.text : sel.extractedText || '';
      return `[Context ${idx + 1} from page ${sel.page}]: ${text}`;
    });

    return contextParts.join('\n\n');
  }, [selectedContext]);

  // Handle form submission
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      if (!isValid || isLoading) return;

      setError(null);
      setIsLoading(true);

      try {
        // Build request with context
        const contextText = buildContextText();
        const payload: ExplainRequest = {
          question: questionTrimmed,
          ...(contextText && { context_text: contextText }),
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
      includeDebug,
      documentId,
      buildContextText,
      onShowAnswer,
    ]
  );

  // Dismiss toast
  const dismissToast = useCallback(() => setToast(null), []);

  // Handle question change
  const handleQuestionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    if (newValue.length <= MAX_QUESTION_LENGTH + 100) {
      setQuestion(newValue);
    }
  };

  return (
    <div className={clsx("space-y-4", className)}>
      {/* Selected Context Display */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="block text-sm font-medium text-gray-700">
            Selected Context
          </label>
          {selectedContext.length > 0 && onClearAllSelections && (
            <button
              onClick={onClearAllSelections}
              className="text-xs text-gray-500 hover:text-red-600 transition-colors"
            >
              Clear all
            </button>
          )}
        </div>

        {selectedContext.length === 0 ? (
          <div className="p-3 bg-gray-50 border border-dashed border-gray-300 rounded-lg text-center">
            <p className="text-sm text-gray-500">
              No context selected
            </p>
            <p className="text-xs text-gray-400 mt-1">
              Select text or draw a region in the PDF to add context
            </p>
          </div>
        ) : (
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {selectedContext.map((selection, index) => (
              <ContextBadge
                key={index}
                selection={selection}
                index={index}
                onRemove={() => onClearSelection?.(index)}
              />
            ))}
          </div>
        )}
      </div>

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
            <>
              {selectedContext.length > 0 && (
                <span className="bg-white/20 px-1.5 py-0.5 rounded text-xs">
                  {selectedContext.length} context{selectedContext.length > 1 ? 's' : ''}
                </span>
              )}
              Ask
            </>
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
