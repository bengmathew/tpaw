const { withSentryConfig } = require('@sentry/nextjs')
const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
})

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config, options) => {
    config.experiments.asyncWebAssembly = true
    return config
  },
  async rewrites() {
    return [
      // Proxy for firebase auth redirects.
      {
        source: '/__/auth/:path*',
        destination: `https://${process.env.NEXT_PUBLIC_GOOGLE_FIREBASE_AUTH_DOMAIN_REDIRECT}/__/auth/:path*`,
      },
    ]
  },
}

// Make sure adding Sentry options is the last code to run before exporting, to
// ensure that your source maps include changes from all other Webpack plugins
module.exports = withSentryConfig(withBundleAnalyzer(nextConfig), {
  hideSourceMaps: false,
  widenClientFileUpload: true,
  tunnelRoute: '/__sentry_tunnel',
})
