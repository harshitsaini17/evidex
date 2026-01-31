"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import clsx from "clsx";

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

export interface PDFViewerProps {
  /** URL to the PDF file (e.g., `${API}/documents/${id}/file`) */
  fileUrl: string;
  /** Page number to highlight and scroll to */
  highlightPage?: number | null;
  /** Callback when PDF loads successfully */
  onLoadSuccess?: (numPages: number) => void;
  /** Additional CSS classes */
  className?: string;
}

export default function PDFViewer({
  fileUrl,
  highlightPage,
  onLoadSuccess,
  className,
}: PDFViewerProps) {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [highlightedPage, setHighlightedPage] = useState<number | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  // Handle successful document load
  const handleDocumentLoadSuccess = useCallback(
    ({ numPages }: { numPages: number }) => {
      setNumPages(numPages);
      setLoading(false);
      onLoadSuccess?.(numPages);
    },
    [onLoadSuccess]
  );

  // Handle document load error
  const handleDocumentLoadError = useCallback((err: Error) => {
    setError(err.message);
    setLoading(false);
  }, []);

  // Navigate to a specific page
  const goToPage = useCallback(
    (page: number) => {
      if (page >= 1 && page <= (numPages || 1)) {
        setCurrentPage(page);
        // Scroll to the page in the container
        const pageElement = pageRefs.current.get(page);
        if (pageElement && containerRef.current) {
          pageElement.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }
    },
    [numPages]
  );

  // Handle page input change
  const handlePageInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value)) {
      goToPage(value);
    }
  };

  // Effect: Scroll to and highlight the specified page
  useEffect(() => {
    if (highlightPage && highlightPage >= 1 && highlightPage <= (numPages || 0)) {
      goToPage(highlightPage);
      // Trigger highlight animation
      setHighlightedPage(highlightPage);
      // Remove highlight after animation completes
      const timer = setTimeout(() => {
        setHighlightedPage(null);
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [highlightPage, numPages, goToPage]);

  // Store ref for each page
  const setPageRef = useCallback((page: number, element: HTMLDivElement | null) => {
    if (element) {
      pageRefs.current.set(page, element);
    } else {
      pageRefs.current.delete(page);
    }
  }, []);

  return (
    <div className={clsx("flex flex-col h-full", className)}>
      {/* Pager Controls */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-100 border-b shrink-0">
        <button
          onClick={() => goToPage(currentPage - 1)}
          disabled={currentPage <= 1 || loading}
          className={clsx(
            "px-3 py-1.5 text-sm font-medium bg-white border rounded-md",
            "hover:bg-gray-50 transition-colors",
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
          aria-label="Previous page"
        >
          Prev
        </button>

        <div className="flex items-center gap-2 text-sm">
          <label htmlFor="page-input" className="sr-only">
            Current page
          </label>
          <input
            id="page-input"
            type="number"
            min={1}
            max={numPages || 1}
            value={currentPage}
            onChange={handlePageInputChange}
            disabled={loading}
            className="w-16 px-2 py-1 text-center border rounded-md"
            aria-label="Current page number"
          />
          <span className="text-gray-600">of {numPages || "?"}</span>
        </div>

        <button
          onClick={() => goToPage(currentPage + 1)}
          disabled={currentPage >= (numPages || 1) || loading}
          className={clsx(
            "px-3 py-1.5 text-sm font-medium bg-white border rounded-md",
            "hover:bg-gray-50 transition-colors",
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
          aria-label="Next page"
        >
          Next
        </button>
      </div>

      {/* PDF Container */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto bg-gray-200 p-4"
        role="document"
        aria-label="PDF document viewer"
      >
        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center h-full">
            <div className="text-gray-600">Loading PDF...</div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="flex items-center justify-center h-full">
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
              <p className="font-medium">Failed to load PDF</p>
              <p className="text-sm mt-1">{error}</p>
              {/* TODO: Add retry button */}
            </div>
          </div>
        )}

        {/* PDF Document */}
        <Document
          file={fileUrl}
          onLoadSuccess={handleDocumentLoadSuccess}
          onLoadError={handleDocumentLoadError}
          loading={null}
          error={null}
          className="flex flex-col items-center gap-4"
        >
          {/* Render all pages for vertical scrolling */}
          {numPages &&
            Array.from({ length: numPages }, (_, index) => {
              const pageNum = index + 1;
              const isHighlighted = highlightedPage === pageNum;

              return (
                <div
                  key={pageNum}
                  ref={(el) => setPageRef(pageNum, el)}
                  className={clsx(
                    "shadow-lg transition-all duration-300",
                    // Highlight animation: flash with a subtle yellow border
                    isHighlighted && "ring-4 ring-yellow-400 animate-pulse"
                  )}
                >
                  <Page
                    pageNumber={pageNum}
                    renderTextLayer={true}
                    renderAnnotationLayer={true}
                    className="bg-white"
                    /*
                     * TODO: Text-to-coordinate highlighting implementation
                     * 
                     * To highlight specific text within a page:
                     * 1. Use the text layer that react-pdf renders (renderTextLayer={true})
                     * 2. After render, query the text layer spans using document.querySelectorAll
                     * 3. Match text content to find the target span(s)
                     * 4. Get bounding box coordinates via getBoundingClientRect()
                     * 5. Overlay a highlight div positioned absolutely over the text
                     * 
                     * Alternatively, use customTextRenderer prop to wrap matched text
                     * in highlight spans during render.
                     * 
                     * For precise paragraph highlighting:
                     * - Backend should provide: { page: number, bbox: [x, y, w, h] }
                     * - Use those coordinates to position highlight overlays
                     * - Account for PDF coordinate system (origin at bottom-left)
                     */
                  />
                </div>
              );
            })}
        </Document>
      </div>
    </div>
  );
}
