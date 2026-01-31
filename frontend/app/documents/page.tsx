"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import useSWR from "swr";
import { fetchDocuments, apiClient } from "@/lib/api";
import { DocumentInfo } from "@/lib/types";
import clsx from "clsx";

export default function DocumentsPage() {
  const { data: documents, error, isLoading, mutate } = useSWR<DocumentInfo[]>(
    "/documents",
    fetchDocuments
  );

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // TODO: improve file upload UX later (drag-drop, progress bar, validation)
    setUploading(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      await apiClient.post("/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // Refresh the document list
      mutate();
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  return (
    <main className="min-h-screen p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Documents</h1>
        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            onClick={handleUploadClick}
            disabled={uploading}
            className={clsx(
              "px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium",
              "hover:bg-blue-700 transition-colors",
              "disabled:opacity-50 disabled:cursor-not-allowed"
            )}
          >
            {uploading ? "Uploading..." : "Upload PDF"}
          </button>
        </div>
      </div>

      {/* Upload Error */}
      {uploadError && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-md text-sm">
          {uploadError}
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <div className="text-gray-500">Loading documents...</div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 text-red-700 rounded-md">
          <p className="font-medium">Failed to load documents</p>
          <p className="text-sm mt-1">{error.message}</p>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && !error && documents?.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          <p>No documents yet.</p>
          <p className="text-sm mt-1">Upload a PDF to get started.</p>
        </div>
      )}

      {/* Document Grid */}
      {!isLoading && !error && documents && documents.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {documents.map((doc) => (
            <DocumentCard key={doc.document_id} document={doc} />
          ))}
        </div>
      )}
    </main>
  );
}

interface DocumentCardProps {
  document: DocumentInfo;
}

function DocumentCard({ document }: DocumentCardProps) {
  const statusColor =
    document.status === "ready"
      ? "bg-green-100 text-green-800"
      : document.status === "processing"
        ? "bg-yellow-100 text-yellow-800"
        : "bg-gray-100 text-gray-800";

  return (
    <div className="p-4 border border-gray-200 rounded-lg bg-white shadow-sm hover:shadow-md transition-shadow">
      <h2 className="font-semibold text-gray-900 truncate" title={document.title}>
        {document.title}
      </h2>
      <div className="mt-2 flex items-center gap-2">
        <span
          className={clsx(
            "px-2 py-0.5 text-xs font-medium rounded-full",
            statusColor
          )}
        >
          {document.status}
        </span>
      </div>
      <div className="mt-4">
        <Link
          href={`/documents/${document.document_id}`}
          className={clsx(
            "inline-block px-3 py-1.5 text-sm font-medium rounded-md",
            "bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors"
          )}
        >
          Open
        </Link>
      </div>
    </div>
  );
}
