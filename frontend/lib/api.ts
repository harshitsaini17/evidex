import axios, { AxiosInstance, AxiosError } from "axios";
import {
  DocumentInfo,
  SectionSummary,
  Paragraph,
  ExplainRequest,
  ExplainResponse,
} from "./types";

/**
 * Base API URL from environment variable
 */
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Axios instance configured for the Evidex API
 */
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30000,
});

/**
 * Extract a readable error message from an Axios error
 */
function extractErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<{
      detail?: string | Array<{ msg?: string; loc?: string[] }>;
      message?: string
    }>;

    const detail = axiosError.response?.data?.detail;
    if (detail) {
      // Handle string detail
      if (typeof detail === 'string') {
        return detail;
      }
      // Handle FastAPI validation error format (array of {msg, loc})
      if (Array.isArray(detail) && detail.length > 0) {
        const firstError = detail[0];
        if (firstError.msg) {
          const location = firstError.loc?.join('.') || '';
          return location ? `${location}: ${firstError.msg}` : firstError.msg;
        }
      }
    }

    if (axiosError.response?.data?.message) {
      return axiosError.response.data.message;
    }
    if (axiosError.response?.status) {
      return `Request failed with status ${axiosError.response.status}`;
    }
    return axiosError.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "An unknown error occurred";
}

/**
 * SWR fetcher wrapper for GET requests
 */
export const fetcher = async <T>(url: string): Promise<T> => {
  try {
    const response = await apiClient.get<T>(url);
    return response.data;
  } catch (error) {
    throw new Error(extractErrorMessage(error));
  }
};

// ============================================================================
// Document API
// ============================================================================

/**
 * Fetch all documents
 */
export async function fetchDocuments(): Promise<DocumentInfo[]> {
  try {
    const response = await apiClient.get<DocumentInfo[]>("/documents");
    return response.data;
  } catch (error) {
    throw new Error(extractErrorMessage(error));
  }
}

/**
 * Fetch sections for a document
 */
export async function fetchSections(documentId: string): Promise<SectionSummary[]> {
  try {
    const response = await apiClient.get<{ document_id: string; sections: SectionSummary[] }>(
      `/documents/${documentId}/sections`
    );
    return response.data.sections;
  } catch (error) {
    throw new Error(extractErrorMessage(error));
  }
}

/**
 * Fetch a single paragraph by ID
 */
export async function fetchParagraph(
  documentId: string,
  paragraphId: string
): Promise<Paragraph> {
  try {
    const response = await apiClient.get<Paragraph>(
      `/documents/${documentId}/paragraphs/${paragraphId}`
    );
    return response.data;
  } catch (error) {
    throw new Error(extractErrorMessage(error));
  }
}

/**
 * Request an explanation for a document
 */
export async function explainDocument(
  documentId: string,
  payload: ExplainRequest
): Promise<ExplainResponse> {
  try {
    const response = await apiClient.post<ExplainResponse>(
      `/documents/${documentId}/explain`,
      payload
    );
    return response.data;
  } catch (error) {
    throw new Error(extractErrorMessage(error));
  }
}
