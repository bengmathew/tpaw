import * as Sentry from '@sentry/node'
import { Config } from './Config.js'

Sentry.init({
  dsn: Config.sentry.dsn,
  integrations: [Sentry.httpIntegration(), Sentry.expressIntegration()],
  tracesSampleRate: 0.01,
})
