"use client";

import clsx from "clsx";
import { SectionSummary } from "@/lib/types";

export interface SectionListProps {
  sections: SectionSummary[];
  selectedSectionId?: string;
  onSelectSection?: (section: SectionSummary) => void;
  className?: string;
}

export default function SectionList({
  sections,
  selectedSectionId,
  onSelectSection,
  className,
}: SectionListProps) {
  // TODO: Implement section list logic
  return (
    <div className={clsx("flex flex-col gap-2", className)}>
      <h3 className="text-lg font-semibold mb-2">Sections</h3>
      {sections.length === 0 ? (
        <p className="text-gray-500 text-sm">No sections available.</p>
      ) : (
        <ul className="space-y-1">
          {sections.map((section) => (
            <li key={section.section_title}>
              <button
                onClick={() => onSelectSection?.(section)}
                className={clsx(
                  "w-full text-left px-3 py-2 rounded transition-colors",
                  selectedSectionId === section.section_title
                    ? "bg-blue-100 text-blue-900"
                    : "hover:bg-gray-100"
                )}
              >
                <span className="font-medium">{section.section_title}</span>
                <span className="text-xs text-gray-500 ml-2">
                  ({section.paragraph_ids.length} paragraphs)
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
