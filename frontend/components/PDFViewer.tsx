"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import clsx from "clsx";
import { TextSelection, RectSelection } from "@/lib/types";

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type SelectionMode = 'none' | 'text' | 'rect';

interface RectDrawState {
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
  page: number;
}

export interface PDFViewerProps {
  /** URL to the PDF file (e.g., `${API}/documents/${id}/file`) */
  fileUrl: string;
  /** Page number to highlight and scroll to */
  highlightPage?: number | null;
  /** Callback when PDF loads successfully */
  onLoadSuccess?: (numPages: number) => void;
  /** Callback when text is selected */
  onTextSelect?: (selection: TextSelection) => void;
  /** Callback when rectangle selection is made */
  onRectSelect?: (selection: RectSelection) => void;
  /** Additional CSS classes */
  className?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Zoom Control Constants
// ─────────────────────────────────────────────────────────────────────────────

const ZOOM_MIN = 0.5;
const ZOOM_MAX = 3.0;
const ZOOM_STEP = 0.25;
const ZOOM_PRESETS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0];

// ─────────────────────────────────────────────────────────────────────────────
// PDFViewer Component
// ─────────────────────────────────────────────────────────────────────────────

export default function PDFViewer({
  fileUrl,
  highlightPage,
  onLoadSuccess,
  onTextSelect,
  onRectSelect,
  className,
}: PDFViewerProps) {
  // Document state
  const [numPages, setNumPages] = useState<number | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [highlightedPage, setHighlightedPage] = useState<number | null>(null);

  // Zoom state
  const [zoom, setZoom] = useState(1.0);

  // Selection mode state
  const [selectionMode, setSelectionMode] = useState<SelectionMode>('text');

  // Rectangle drawing state
  const [rectDraw, setRectDraw] = useState<RectDrawState | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // Refs
  const containerRef = useRef<HTMLDivElement>(null);
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  // ─────────────────────────────────────────────────────────────────────────
  // Document Loading
  // ─────────────────────────────────────────────────────────────────────────

  const handleDocumentLoadSuccess = useCallback(
    ({ numPages }: { numPages: number }) => {
      setNumPages(numPages);
      setLoading(false);
      onLoadSuccess?.(numPages);
    },
    [onLoadSuccess]
  );

  const handleDocumentLoadError = useCallback((err: Error) => {
    setError(err.message);
    setLoading(false);
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // Navigation
  // ─────────────────────────────────────────────────────────────────────────

  const goToPage = useCallback(
    (page: number) => {
      if (page >= 1 && page <= (numPages || 1)) {
        setCurrentPage(page);
        const pageElement = pageRefs.current.get(page);
        if (pageElement && containerRef.current) {
          pageElement.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }
    },
    [numPages]
  );

  const handlePageInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value)) {
      goToPage(value);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Zoom Controls
  // ─────────────────────────────────────────────────────────────────────────

  const zoomIn = useCallback(() => {
    setZoom((prev) => Math.min(prev + ZOOM_STEP, ZOOM_MAX));
  }, []);

  const zoomOut = useCallback(() => {
    setZoom((prev) => Math.max(prev - ZOOM_STEP, ZOOM_MIN));
  }, []);

  const resetZoom = useCallback(() => {
    setZoom(1.0);
  }, []);

  const setZoomLevel = useCallback((level: number) => {
    setZoom(Math.max(ZOOM_MIN, Math.min(level, ZOOM_MAX)));
  }, []);

  // Keyboard shortcuts for zoom
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === '=') {
        e.preventDefault();
        zoomIn();
      } else if ((e.ctrlKey || e.metaKey) && e.key === '-') {
        e.preventDefault();
        zoomOut();
      } else if ((e.ctrlKey || e.metaKey) && e.key === '0') {
        e.preventDefault();
        resetZoom();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [zoomIn, zoomOut, resetZoom]);

  // ─────────────────────────────────────────────────────────────────────────
  // Text Selection Handler
  // ─────────────────────────────────────────────────────────────────────────

  const handleTextSelection = useCallback(() => {
    if (selectionMode !== 'text') return;

    const selection = window.getSelection();
    if (!selection || selection.isCollapsed) return;

    const selectedText = selection.toString().trim();
    if (!selectedText) return;

    // Find which page the selection is in
    const anchorNode = selection.anchorNode;
    if (!anchorNode) return;

    let pageNumber = 1;
    let element = anchorNode.parentElement;
    while (element) {
      const pageAttr = element.getAttribute('data-page-number');
      if (pageAttr) {
        pageNumber = parseInt(pageAttr, 10);
        break;
      }
      element = element.parentElement;
    }

    onTextSelect?.({
      type: 'text',
      text: selectedText,
      page: pageNumber,
    });

    // Clear the selection after capturing
    selection.removeAllRanges();
  }, [selectionMode, onTextSelect]);

  // Listen for mouse up events to capture text selection
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('mouseup', handleTextSelection);
    return () => container.removeEventListener('mouseup', handleTextSelection);
  }, [handleTextSelection]);

  // ─────────────────────────────────────────────────────────────────────────
  // Rectangle Selection Handler
  // ─────────────────────────────────────────────────────────────────────────

  const handleMouseDown = useCallback(
    (e: React.MouseEvent, pageNum: number) => {
      if (selectionMode !== 'rect') return;

      const pageElement = pageRefs.current.get(pageNum);
      if (!pageElement) return;

      const rect = pageElement.getBoundingClientRect();
      const x = (e.clientX - rect.left) / zoom;
      const y = (e.clientY - rect.top) / zoom;

      setIsDrawing(true);
      setRectDraw({
        startX: x,
        startY: y,
        currentX: x,
        currentY: y,
        page: pageNum,
      });
    },
    [selectionMode, zoom]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent, pageNum: number) => {
      if (!isDrawing || !rectDraw || rectDraw.page !== pageNum) return;

      const pageElement = pageRefs.current.get(pageNum);
      if (!pageElement) return;

      const rect = pageElement.getBoundingClientRect();
      const x = (e.clientX - rect.left) / zoom;
      const y = (e.clientY - rect.top) / zoom;

      setRectDraw((prev) =>
        prev ? { ...prev, currentX: x, currentY: y } : null
      );
    },
    [isDrawing, rectDraw, zoom]
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent, pageNum: number) => {
      if (!isDrawing || !rectDraw || rectDraw.page !== pageNum) {
        setIsDrawing(false);
        setRectDraw(null);
        return;
      }

      const pageElement = pageRefs.current.get(pageNum);
      if (!pageElement) {
        setIsDrawing(false);
        setRectDraw(null);
        return;
      }

      const rect = pageElement.getBoundingClientRect();
      const endX = (e.clientX - rect.left) / zoom;
      const endY = (e.clientY - rect.top) / zoom;

      // Calculate normalized bbox
      const x = Math.min(rectDraw.startX, endX);
      const y = Math.min(rectDraw.startY, endY);
      const width = Math.abs(endX - rectDraw.startX);
      const height = Math.abs(endY - rectDraw.startY);

      // Only emit if rectangle is meaningful size
      if (width > 10 && height > 10) {
        // Try to extract text from the selected region
        const textLayer = pageElement.querySelector('.react-pdf__Page__textContent');
        let extractedText = '';

        if (textLayer) {
          const spans = textLayer.querySelectorAll('span');
          spans.forEach((span) => {
            const spanRect = span.getBoundingClientRect();
            const spanX = (spanRect.left - rect.left) / zoom;
            const spanY = (spanRect.top - rect.top) / zoom;
            const spanRight = spanX + spanRect.width / zoom;
            const spanBottom = spanY + spanRect.height / zoom;

            // Check if span overlaps with selection rectangle
            if (
              spanX < x + width &&
              spanRight > x &&
              spanY < y + height &&
              spanBottom > y
            ) {
              extractedText += (span.textContent || '') + ' ';
            }
          });
        }

        onRectSelect?.({
          type: 'rect',
          page: pageNum,
          bbox: { x, y, width, height },
          extractedText: extractedText.trim() || undefined,
        });
      }

      setIsDrawing(false);
      setRectDraw(null);
    },
    [isDrawing, rectDraw, zoom, onRectSelect]
  );

  // ─────────────────────────────────────────────────────────────────────────
  // Highlight Page Effect
  // ─────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (highlightPage && highlightPage >= 1 && highlightPage <= (numPages || 0)) {
      goToPage(highlightPage);
      setHighlightedPage(highlightPage);
      const timer = setTimeout(() => {
        setHighlightedPage(null);
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [highlightPage, numPages, goToPage]);

  // ─────────────────────────────────────────────────────────────────────────
  // Page Ref Management
  // ─────────────────────────────────────────────────────────────────────────

  const setPageRef = useCallback((page: number, element: HTMLDivElement | null) => {
    if (element) {
      pageRefs.current.set(page, element);
    } else {
      pageRefs.current.delete(page);
    }
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className={clsx("flex flex-col h-full", className)}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-100 border-b shrink-0 gap-2 flex-wrap">
        {/* Page Navigation */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => goToPage(currentPage - 1)}
            disabled={currentPage <= 1 || loading}
            className={clsx(
              "px-2 py-1 text-sm font-medium bg-white border rounded-md",
              "hover:bg-gray-50 transition-colors",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
            aria-label="Previous page"
          >
            ←
          </button>

          <div className="flex items-center gap-1 text-sm">
            <input
              type="number"
              min={1}
              max={numPages || 1}
              value={currentPage}
              onChange={handlePageInputChange}
              disabled={loading}
              className="w-12 px-1 py-1 text-center border rounded-md text-sm"
              aria-label="Current page number"
            />
            <span className="text-gray-600">/ {numPages || "?"}</span>
          </div>

          <button
            onClick={() => goToPage(currentPage + 1)}
            disabled={currentPage >= (numPages || 1) || loading}
            className={clsx(
              "px-2 py-1 text-sm font-medium bg-white border rounded-md",
              "hover:bg-gray-50 transition-colors",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
            aria-label="Next page"
          >
            →
          </button>
        </div>

        {/* Zoom Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={zoomOut}
            disabled={zoom <= ZOOM_MIN}
            className={clsx(
              "w-8 h-8 flex items-center justify-center text-lg font-bold",
              "bg-white border rounded-md hover:bg-gray-50 transition-colors",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
            aria-label="Zoom out"
            title="Zoom out (Ctrl+-)"
          >
            −
          </button>

          <select
            value={zoom}
            onChange={(e) => setZoomLevel(parseFloat(e.target.value))}
            className="px-2 py-1 border rounded-md text-sm bg-white"
            aria-label="Zoom level"
          >
            {ZOOM_PRESETS.map((level) => (
              <option key={level} value={level}>
                {Math.round(level * 100)}%
              </option>
            ))}
          </select>

          <button
            onClick={zoomIn}
            disabled={zoom >= ZOOM_MAX}
            className={clsx(
              "w-8 h-8 flex items-center justify-center text-lg font-bold",
              "bg-white border rounded-md hover:bg-gray-50 transition-colors",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
            aria-label="Zoom in"
            title="Zoom in (Ctrl++)"
          >
            +
          </button>

          <button
            onClick={resetZoom}
            className={clsx(
              "px-2 py-1 text-sm bg-white border rounded-md",
              "hover:bg-gray-50 transition-colors"
            )}
            aria-label="Reset zoom"
            title="Reset zoom (Ctrl+0)"
          >
            Reset
          </button>
        </div>

        {/* Selection Mode Toggle */}
        <div className="flex items-center gap-1 bg-white border rounded-md p-0.5">
          <button
            onClick={() => setSelectionMode('text')}
            className={clsx(
              "px-3 py-1 text-sm rounded transition-colors",
              selectionMode === 'text'
                ? "bg-blue-600 text-white"
                : "text-gray-600 hover:bg-gray-100"
            )}
            aria-label="Text selection mode"
            title="Select text"
          >
            <span className="flex items-center gap-1">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
              </svg>
              Text
            </span>
          </button>
          <button
            onClick={() => setSelectionMode('rect')}
            className={clsx(
              "px-3 py-1 text-sm rounded transition-colors",
              selectionMode === 'rect'
                ? "bg-blue-600 text-white"
                : "text-gray-600 hover:bg-gray-100"
            )}
            aria-label="Rectangle selection mode"
            title="Draw rectangle to select region"
          >
            <span className="flex items-center gap-1">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth={2} />
              </svg>
              Region
            </span>
          </button>
        </div>
      </div>

      {/* PDF Container */}
      <div
        ref={containerRef}
        className={clsx(
          "flex-1 overflow-auto bg-gray-200 p-4",
          selectionMode === 'rect' && "cursor-crosshair"
        )}
        role="document"
        aria-label="PDF document viewer"
      >
        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center h-full">
            <div className="flex flex-col items-center gap-2">
              <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
              <span className="text-gray-600">Loading PDF...</span>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="flex items-center justify-center h-full">
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
              <p className="font-medium">Failed to load PDF</p>
              <p className="text-sm mt-1">{error}</p>
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
          {numPages &&
            Array.from({ length: numPages }, (_, index) => {
              const pageNum = index + 1;
              const isHighlighted = highlightedPage === pageNum;

              return (
                <div
                  key={pageNum}
                  ref={(el) => setPageRef(pageNum, el)}
                  data-page-number={pageNum}
                  className={clsx(
                    "shadow-lg transition-all duration-300 relative",
                    isHighlighted && "ring-4 ring-yellow-400 animate-pulse",
                    selectionMode === 'rect' && "select-none"
                  )}
                  onMouseDown={(e) => handleMouseDown(e, pageNum)}
                  onMouseMove={(e) => handleMouseMove(e, pageNum)}
                  onMouseUp={(e) => handleMouseUp(e, pageNum)}
                  onMouseLeave={(e) => {
                    if (isDrawing && rectDraw?.page === pageNum) {
                      handleMouseUp(e, pageNum);
                    }
                  }}
                >
                  <Page
                    pageNumber={pageNum}
                    scale={zoom}
                    renderTextLayer={true}
                    renderAnnotationLayer={true}
                    className="bg-white"
                  />

                  {/* Rectangle selection overlay */}
                  {isDrawing && rectDraw && rectDraw.page === pageNum && (
                    <div
                      className="absolute border-2 border-blue-500 bg-blue-500/10 pointer-events-none"
                      style={{
                        left: Math.min(rectDraw.startX, rectDraw.currentX) * zoom,
                        top: Math.min(rectDraw.startY, rectDraw.currentY) * zoom,
                        width: Math.abs(rectDraw.currentX - rectDraw.startX) * zoom,
                        height: Math.abs(rectDraw.currentY - rectDraw.startY) * zoom,
                      }}
                    />
                  )}

                  {/* Page number badge */}
                  <div className="absolute bottom-2 right-2 px-2 py-1 bg-black/50 text-white text-xs rounded">
                    Page {pageNum}
                  </div>
                </div>
              );
            })}
        </Document>
      </div>

      {/* Selection Mode Hint */}
      <div className="px-4 py-2 bg-gray-50 border-t text-xs text-gray-500 text-center">
        {selectionMode === 'text' && (
          <span>💡 Select text in the PDF to add it as context for your question</span>
        )}
        {selectionMode === 'rect' && (
          <span>💡 Click and drag to select a region in the PDF</span>
        )}
      </div>
    </div>
  );
}
