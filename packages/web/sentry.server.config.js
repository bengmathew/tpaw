import * as Sentry from '@sentry/nextjs'

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  maxValueLength: 10000,
  // We recommend adjusting this value in production, or using tracesSampler
  // for finer control
  tracesSampleRate: 1.0,
  // integrations will be all default integrations
  integrations: (integrations) =>
    integrations.filter((integration) => integration.name != 'Dedupe'),
  // ...
  // Note: if you want to override the automatic release value, do not set a
  // `release` value here - use the environment variable `SENTRY_RELEASE`, so
  // that it will also get attached to your source maps
  beforeSend(event) {
    if (event.request?.url) {
      const url = new URL(event.request.url)
      if (url.searchParams.get('params')) {
        url.searchParams.set('params', 'redacted')
        event.request.url = url.toString()
      }
    }
    if (event.request?.query_string) {
      const query = new URLSearchParams(event.release.query_string)
      if (query.get('params')) {
        query.set('params', 'redacted')
        event.request.query_string = query.toString()
      }
    }
    return event
  },
})
