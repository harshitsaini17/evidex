# Evidex Frontend

A Next.js (App Router) TypeScript frontend for the Evidex document analysis platform.

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Environment Setup

Copy the example environment file and configure your API URL:

```bash
cp .env.local.example .env.local
```

Edit `.env.local` to set `NEXT_PUBLIC_API_URL` to your backend API endpoint.

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm run start
```

### Linting & Formatting

```bash
npm run lint        # Check for linting errors
npm run lint:fix    # Auto-fix linting errors
npm run format      # Format with Prettier
npm run format:check # Check formatting
```

### Testing

```bash
npm run test        # Run tests
npm run test:watch  # Run tests in watch mode
```

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx           # Root layout
│   ├── page.tsx             # Home page
│   └── documents/
│       ├── page.tsx         # Document list page
│       └── [id]/
│           └── page.tsx     # Document reader page
├── components/
│   ├── PDFViewer.tsx        # PDF rendering component
│   ├── SectionList.tsx      # Document sections list
│   ├── ParagraphItem.tsx    # Individual paragraph display
│   ├── QuestionPanel.tsx    # Q&A input panel
│   └── AnswerCard.tsx       # Answer display card
├── lib/
│   ├── api.ts               # Axios instance & API helpers
│   └── types.ts             # Shared TypeScript types
├── styles/
│   └── globals.css          # Tailwind CSS imports
└── ...config files
```

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **HTTP Client**: Axios
- **Data Fetching**: SWR
- **PDF Rendering**: react-pdf
- **Linting**: ESLint + Prettier
