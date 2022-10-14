const {withSentryConfig} = require('@sentry/nextjs')
// const SSRPlugin = require("next/dist/build/webpack/plugins/nextjs-ssr-import")
//   .default;
//  const { dirname, relative, resolve, join } = require("path");

/** @type {import('next').NextConfig} */
const moduleExports = {
  reactStrictMode: true,
  sentry: {
    hideSourceMaps: true,
  },
  // https://github.com/vercel/next.js/issues/25484#issuecomment-874942566
  webpack: (config, options) => {
    // if (!config.optimization.splitChunks.cacheGroups) {
    //   config.optimization.splitChunks.cacheGroups = {
    //     defaultVendors: false,
    //     default: false,
    //     framework: {
    //       chunks: 'all',
    //       name: 'framework',
    //       test: /(?<!node_modules.*)[\\/]node_modules[\\/](react|react-dom|scheduler|prop-types|use-subscription)[\\/]/,
    //       priority: 40,
    //       enforce: true,
    //     },
    //     commons: {
    //       name: 'commons',
    //       chunks: 'initial',
    //       minChunks: 20,
    //       priority: 20,
    //     },
    //   }
    // }

    config.experiments.asyncWebAssembly = true
    // config.experiments.syncWebAssembly = true
    // const ssrPlugin = config.plugins.find(
    //   plugin => plugin instanceof SSRPlugin
    // );

    // if (ssrPlugin) {
    //   patchSsrPlugin(ssrPlugin);
    // }

    return config
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
module.exports = withSentryConfig(moduleExports, SentryWebpackPluginOptions)
