"use client";

import { useParams } from "next/navigation";
import useSWR from "swr";
import Link from "next/link";
import clsx from "clsx";
import { fetchSections } from "@/lib/api";
import { SectionSummary } from "@/lib/types";
import PDFViewer from "@/components/PDFViewer";
import SectionList from "@/components/SectionList";
import QuestionPanel from "@/components/QuestionPanel";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function DocumentReaderPage() {
  const params = useParams();
  const documentId = params.id as string;

  // Fetch sections for this document
  const {
    data: sections,
    error: sectionsError,
    isLoading: sectionsLoading,
  } = useSWR<SectionSummary[]>(
    documentId ? `/documents/${documentId}/sections` : null,
    () => fetchSections(documentId)
  );

  // PDF file URL
  // TODO: Confirm backend route for PDF file retrieval
  const pdfUrl = `${API_BASE_URL}/documents/${documentId}/file`;

  const handlePdfLoadSuccess = (numPages: number) => {
    console.log(`PDF loaded with ${numPages} pages`);
    // TODO: Could store numPages in state for UI display
  };

  const handleSectionSelect = (section: SectionSummary) => {
    console.log("Selected section:", section.section_title);
    // TODO: Implement scroll to section / highlight paragraphs
  };

  const handleAskQuestion = (question: string) => {
    console.log("Question asked:", question);
    // TODO: Implement Q&A submission logic
  };

  // Error state: document or sections not found
  if (sectionsError) {
    return (
      <main className="min-h-screen p-8">
        <div className="max-w-2xl mx-auto">
          <div className="p-6 bg-red-50 border border-red-200 rounded-lg">
            <h1 className="text-xl font-semibold text-red-800">
              Document Not Found
            </h1>
            <p className="mt-2 text-red-700">
              {sectionsError.message || "Unable to load this document."}
            </p>
            <Link
              href="/documents"
              className="inline-block mt-4 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
            >
              Back to Documents
            </Link>
          </div>
        </div>
      </main>
    );
  }

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
          <h1 className="text-lg font-semibold">Document: {documentId}</h1>
        </div>
      </header>

      {/* Main content: two-column layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel: PDF viewer (70% width) */}
        <section className="w-[70%] border-r flex flex-col bg-gray-50">
          <PDFViewer
            fileUrl={pdfUrl}
            onLoadSuccess={handlePdfLoadSuccess}
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
            {sectionsLoading ? (
              <p className="text-sm text-gray-500">Loading sections...</p>
            ) : (
              <SectionList
                sections={sections || []}
                onSelectSection={handleSectionSelect}
              />
            )}
          </div>

          {/* Question Panel */}
          <div className="p-4">
            <QuestionPanel
              documentId={documentId}
              onAsk={handleAskQuestion}
            />
          </div>
        </aside>
      </div>
    </main>
  );
}
