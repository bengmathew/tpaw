const { withSentryConfig } = require('@sentry/nextjs')
const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
})

// const SSRPlugin = require("next/dist/build/webpack/plugins/nextjs-ssr-import")
//   .default;
//  const { dirname, relative, resolve, join } = require("path");

/** @type {import('next').NextConfig} */
const moduleExports = {
  reactStrictMode: true,
  sentry: {
    hideSourceMaps: false,
    widenClientFileUpload: true,
  },
  webpack: (config, options) => {
    config.experiments.asyncWebAssembly = true
    // const ssrPlugin = config.plugins.find(
    //   plugin => plugin instanceof SSRPlugin
    // );

    // if (ssrPlugin) {
    //   patchSsrPlugin(ssrPlugin);
    // }

    return config
  },
  // Option 3 (proxy) for firebase auth redirects.
  async rewrites() {
    return [
      {
        source: '/__/auth/:path*',
        destination: `https://${process.env.NEXT_PUBLIC_GOOGLE_FIREBASE_AUTH_DOMAIN_REDIRECT}/__/auth/:path*`,
      },
    ]
  },
}

const SentryWebpackPluginOptions = {
  // Additional config options for the Sentry Webpack plugin. Keep in mind that
  // the following options are set automatically, and overriding them is not
  // recommended:
  //   release, url, org, project, authToken, configFile, stripPrefix,
  //   urlPrefix, include, ignore

  silent: true, // Suppresses all logs
  // For all available options, see:
  // https://github.com/getsentry/sentry-webpack-plugin#options.
}

// Make sure adding Sentry options is the last code to run before exporting, to
// ensure that your source maps include changes from all other Webpack plugins
module.exports = withSentryConfig(
  withBundleAnalyzer(moduleExports, SentryWebpackPluginOptions),
)
