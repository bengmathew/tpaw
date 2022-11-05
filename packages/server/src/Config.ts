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
  },
  postmark: {
    apiToken: fGet(process.env['POSTMARK_API_TOKEN']),
  },
  sentry: {
    dsn: fGet(process.env['SENTRY_DSN']),
  },
}
