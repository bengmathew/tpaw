import dotenv from 'dotenv'
import { fGet } from './Utils/Utils.js'

dotenv.config()
export const Config = {
  port: fGet(process.env['PORT']),
  websiteURL: fGet(process.env['WEBSITE_URL']),
  google: {
    sql: {
      // Used by prisma client.
      databaseURL: fGet(process.env['DATABASE_URL']),
    },
    // Only needed in non Google hosted environments. This should be path to
    // JSON key file.
    applicationDefaultCredentials:
      process.env['GOOGLE_APPLICATION_CREDENTIALS'],
    marketDataBucket: fGet(process.env['MARKET_DATA_BUCKET']),
  },
  postmark: {
    apiToken: fGet(process.env['POSTMARK_API_TOKEN']),
  },
  sentry: {
    dsn: fGet(process.env['SENTRY_DSN']),
  },
  eod: {
    apiKey: fGet(process.env['EOD_API_KEY']),
  },
  stLouisFed: {
    apiKey: fGet(process.env['ST_LOUIS_FED_API_KEY']),
  },
  deployFrontEnd: {
    token: fGet(process.env['DEPLOY_FRONTEND_TOKEN']),
    url: fGet(process.env['DEPLOY_FRONTEND_URL']),
  },
}
