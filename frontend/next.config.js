/** @type {import('next').NextConfig} */
const nextConfig = {
  // Turbopack configuration (Next.js 16+ default)
  turbopack: {
    resolveAlias: {
      // Required for react-pdf to work properly - use empty object to exclude
      canvas: { browser: "" },
    },
  },
  // Webpack fallback for non-Turbopack builds
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    return config;
  },
};

module.exports = nextConfig;
