"use client";

import { useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import PDFViewer from "@/components/PDFViewer";
import SectionList from "@/components/SectionList";
import QuestionPanel from "@/components/QuestionPanel";
import { ContextSelection, TextSelection, RectSelection } from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function DocumentReaderPage() {
  const params = useParams();
  const documentId = params.id as string;

  // PDF file URL
  const pdfUrl = `${API_BASE_URL}/documents/${documentId}/file`;

  // Track highlighted page for PDF viewer
  const [highlightPage, setHighlightPage] = useState<number | null>(null);

  // Track selected context for Q&A
  const [selectedContext, setSelectedContext] = useState<ContextSelection[]>([]);

  const handlePdfLoadSuccess = (numPages: number) => {
    console.log(`PDF loaded with ${numPages} pages`);
  };

  // Handle paragraph selection from SectionList
  const handleParagraphSelect = useCallback((paragraphId: string, page?: number) => {
    console.log("Selected paragraph:", paragraphId, "page:", page);
    if (page) {
      setHighlightPage(page);
    }
  }, []);

  // Handle citation click from QuestionPanel - scroll to paragraph
  const handleCitationClick = useCallback((citationId: string) => {
    console.log("Citation clicked:", citationId);
    // For now, just log - could scroll to paragraph in future
  }, []);

  // Handle text selection from PDF viewer
  const handleTextSelect = useCallback((selection: TextSelection) => {
    setSelectedContext((prev) => [...prev, selection]);
  }, []);

  // Handle rectangle selection from PDF viewer
  const handleRectSelect = useCallback((selection: RectSelection) => {
    // Only add if there's extracted text
    if (selection.extractedText) {
      setSelectedContext((prev) => [...prev, selection]);
    }
  }, []);

  // Clear a specific selection
  const handleClearSelection = useCallback((index: number) => {
    setSelectedContext((prev) => prev.filter((_, i) => i !== index));
  }, []);

  // Clear all selections
  const handleClearAllSelections = useCallback(() => {
    setSelectedContext([]);
  }, []);

  return (
    <main className="h-screen flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 border-b bg-white">
        <div className="flex items-center gap-4">
          <Link
            href="/documents"
            className="text-sm text-gray-600 hover:text-gray-900"
          >
            ← Back
          </Link>
          <h1 className="text-lg font-semibold truncate" title={documentId}>
            Document: {documentId.slice(0, 8)}...
          </h1>
        </div>

        {/* Context indicator */}
        {selectedContext.length > 0 && (
          <div className="flex items-center gap-2 text-sm">
            <span className="text-gray-600">
              {selectedContext.length} selection{selectedContext.length > 1 ? 's' : ''}
            </span>
            <button
              onClick={handleClearAllSelections}
              className="text-red-500 hover:text-red-700 transition-colors"
            >
              Clear all
            </button>
          </div>
        )}
      </header>

      {/* Main content: two-column layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel: PDF viewer (70% width) */}
        <section className="w-[70%] border-r flex flex-col bg-gray-50">
          <PDFViewer
            fileUrl={pdfUrl}
            highlightPage={highlightPage}
            onLoadSuccess={handlePdfLoadSuccess}
            onTextSelect={handleTextSelect}
            onRectSelect={handleRectSelect}
            className="flex-1"
          />
        </section>

        {/* Right panel: Sections and Q&A (30% width) */}
        <aside className="w-[30%] flex flex-col overflow-hidden bg-white">
          {/* Sections */}
          <div className="flex-1 overflow-y-auto p-4 border-b">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Sections
            </h2>
            <SectionList
              documentId={documentId}
              onParagraphSelect={handleParagraphSelect}
            />
          </div>

          {/* Question Panel */}
          <div className="p-4 overflow-y-auto max-h-[50%]">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
              Ask a Question
            </h2>
            <QuestionPanel
              documentId={documentId}
              selectedContext={selectedContext}
              onClearSelection={handleClearSelection}
              onClearAllSelections={handleClearAllSelections}
              onCitationClick={handleCitationClick}
            />
          </div>
        </aside>
      </div>
    </main>
  );
}
