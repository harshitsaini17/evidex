"use client";

import clsx from "clsx";
import { Paragraph } from "@/lib/types";

export interface ParagraphItemProps {
  paragraph: Paragraph;
  isSelected?: boolean;
  onClick?: (paragraph: Paragraph) => void;
  className?: string;
}

export default function ParagraphItem({
  paragraph,
  isSelected,
  onClick,
  className,
}: ParagraphItemProps) {
  // TODO: Implement paragraph item logic
  return (
    <div
      onClick={() => onClick?.(paragraph)}
      className={clsx(
        "p-3 border rounded cursor-pointer transition-colors",
        isSelected ? "border-blue-500 bg-blue-50" : "border-gray-200 hover:border-gray-300",
        className
      )}
    >
      <div className="flex items-start justify-between mb-1">
        <span className="text-xs text-gray-500">{paragraph.paragraph_id}</span>
        {paragraph.page && <span className="text-xs text-gray-400">Page {paragraph.page}</span>}
      </div>
      <p className="text-sm text-gray-700 line-clamp-3">{paragraph.text || ""}</p>
    </div>
  );
}
