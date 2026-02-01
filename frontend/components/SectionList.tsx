"use client";

import { useState, useCallback } from "react";
import useSWR, { mutate } from "swr";
import clsx from "clsx";
import { fetchSections, fetchParagraph } from "@/lib/api";
import { SectionSummary, Paragraph } from "@/lib/types";
import ParagraphItem from "./ParagraphItem";

export interface SectionListProps {
  documentId: string;
  onParagraphSelect: (paragraphId: string, page?: number) => void;
  className?: string;
}

// Cache key generator for paragraphs
const getParagraphKey = (documentId: string, paragraphId: string) =>
  `/documents/${documentId}/paragraphs/${paragraphId}`;

export default function SectionList({
  documentId,
  onParagraphSelect,
  className,
}: SectionListProps) {
  // Fetch sections
  const {
    data: sections,
    error,
    isLoading,
  } = useSWR<SectionSummary[]>(
    documentId ? `/documents/${documentId}/sections` : null,
    () => fetchSections(documentId)
  );

  // Track expanded sections
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());
  
  // Track selected paragraph
  const [selectedParagraphId, setSelectedParagraphId] = useState<string | null>(null);
  
  // Track loaded paragraphs (local cache)
  const [paragraphCache, setParagraphCache] = useState<Map<string, Paragraph>>(new Map());
  
  // Track loading paragraphs
  const [loadingParagraphs, setLoadingParagraphs] = useState<Set<string>>(new Set());

  // Toggle section expansion
  const toggleSection = useCallback((sectionTitle: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(sectionTitle)) {
        next.delete(sectionTitle);
      } else {
        next.add(sectionTitle);
      }
      return next;
    });
  }, []);

  // Handle paragraph click
  const handleParagraphClick = useCallback(
    async (paragraphId: string) => {
      setSelectedParagraphId(paragraphId);

      // Check if already cached
      const cached = paragraphCache.get(paragraphId);
      if (cached) {
        onParagraphSelect(paragraphId, cached.page);
        return;
      }

      // Fetch paragraph
      setLoadingParagraphs((prev) => new Set(prev).add(paragraphId));
      
      try {
        const paragraph = await fetchParagraph(documentId, paragraphId);
        
        // Cache locally
        setParagraphCache((prev) => new Map(prev).set(paragraphId, paragraph));
        
        // Also cache in SWR
        mutate(getParagraphKey(documentId, paragraphId), paragraph, false);
        
        onParagraphSelect(paragraphId, paragraph.page);
      } catch (err) {
        console.error("Failed to fetch paragraph:", err);
      } finally {
        setLoadingParagraphs((prev) => {
          const next = new Set(prev);
          next.delete(paragraphId);
          return next;
        });
      }
    },
    [documentId, paragraphCache, onParagraphSelect]
  );

  // Loading state
  if (isLoading) {
    return (
      <div className={clsx("flex flex-col gap-2", className)}>
        <p className="text-gray-500 text-sm">Loading sections...</p>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className={clsx("flex flex-col gap-2", className)}>
        <p className="text-red-500 text-sm">Failed to load sections</p>
        <p className="text-red-400 text-xs">{error.message}</p>
      </div>
    );
  }

  // Empty state
  if (!sections || sections.length === 0) {
    return (
      <div className={clsx("flex flex-col gap-2", className)}>
        <p className="text-gray-500 text-sm">No sections available.</p>
      </div>
    );
  }

  return (
    <div className={clsx("flex flex-col gap-1", className)}>
      {sections.map((section) => {
        const isExpanded = expandedSections.has(section.title);
        
        return (
          <div key={section.title} className="border rounded-lg overflow-hidden">
            {/* Section Header (Accordion Toggle) */}
            <button
              onClick={() => toggleSection(section.title)}
              className={clsx(
                "w-full flex items-center justify-between px-3 py-2 text-left",
                "bg-gray-50 hover:bg-gray-100 transition-colors",
                "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-inset"
              )}
              aria-expanded={isExpanded}
              aria-controls={`section-${section.title}`}
            >
              <span className="font-medium text-sm text-gray-900 truncate">
                {section.title}
              </span>
              <span className="flex items-center gap-2 shrink-0">
                <span className="text-xs text-gray-500">
                  {section.paragraph_ids.length} ¶
                </span>
                <svg
                  className={clsx(
                    "w-4 h-4 text-gray-500 transition-transform",
                    isExpanded && "rotate-180"
                  )}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </span>
            </button>

            {/* Section Content (Paragraphs) */}
            {isExpanded && (
              <div
                id={`section-${section.title}`}
                className="border-t bg-white"
                role="region"
                aria-label={`Paragraphs in ${section.title}`}
              >
                <ul className="divide-y divide-gray-100">
                  {section.paragraph_ids.map((paragraphId) => {
                    const cachedParagraph = paragraphCache.get(paragraphId);
                    const isSelected = selectedParagraphId === paragraphId;
                    const isLoading = loadingParagraphs.has(paragraphId);

                    return (
                      <li key={paragraphId}>
                        <ParagraphItem
                          paragraphId={paragraphId}
                          paragraph={cachedParagraph}
                          isSelected={isSelected}
                          isLoading={isLoading}
                          onSelect={() => handleParagraphClick(paragraphId)}
                        />
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
