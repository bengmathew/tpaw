import dotenv from 'dotenv'
import { fGet } from './Utils/Utils.js'

dotenv.config()
export const Config = {
  port: fGet(process.env['PORT']),
  isProduction: process.env['NODE_ENV'] === 'production',
  interServerToken: fGet(process.env['INTER_SERVER_TOKEN']),
  sentry: {
    dsn: fGet(process.env['SENTRY_DSN_PRINTSERVER']),
  },
  test: {
    url: new URL(fGet(process.env['TEST_URL'])),
    auth: process.env['TEST_URL_AUTH'] ?? null,
  },
}
