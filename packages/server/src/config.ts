import { fGet } from './Utils/Utils.js'

export const Config = {
  port: fGet(process.env['PORT']),
  websiteURL: fGet(process.env['WEBSITE_URL']),
  postmark: {
    apiToken: fGet(process.env['POSTMARK_API_TOKEN']),
  },
  sentry: {
    dsn: fGet(process.env['SENTRY_DSN']),
  },
}
