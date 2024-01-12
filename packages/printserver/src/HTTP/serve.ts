import Sentry from '@sentry/node'
import bodyParser from 'body-parser'
import express from 'express'
import {
  JSONGuard,
  chain,
  gte,
  integer,
  nullable,
  number,
  object,
  string,
} from 'json-guard'
import morgan from 'morgan'
import { setTimeout } from 'timers/promises'
import { cli } from '../CLI/CLI.js'
import { Config } from '../Config.js'
import { assert, block } from '../Utils/Utils.js'
// import { getPageLoadStatus, pagePool } from './PagePool.js'
import asyncHandler from 'express-async-handler'
import { Page } from 'puppeteer'
import { browser } from './browser.js'

cli.command('serve').action(async () => await _impl())
const TIMEOUT = 60 * 1000

async function _impl() {
  const server = express()

  Sentry.init({
    dsn: Config.sentry.dsn,
    integrations: [
      new Sentry.Integrations.Http({ tracing: true }),
      new Sentry.Integrations.Express({ app: server }),
    ],
    tracesSampleRate: 0.01,
  })

  // RequestHandler creates a separate execution context, so that all
  // transactions/spans/breadcrumbs are isolated across requests
  server.use(Sentry.Handlers.requestHandler())
  // TracingHandler creates a trace for every incoming request
  server.use(Sentry.Handlers.tracingHandler())

  if (!Config.isProduction) server.use(morgan('tiny'))

  server.get('/', (req, res) => res.send('I am root!'))
  server.get('/ping', (req, res) => res.send('pong'))
  server.get('/which', (req, res) => {
    const lines = [`Version: 3`, `NODE_ENV: ${process.env['NODE_ENV']}`]
    res.set('Content-Type', 'text/plain')
    res.send(lines.join('\n'))
  })
  server.get('/crash', () => {
    throw new Error('crash')
  })

  server.get('/time', (req, res) => res.send(`${Date.now()}`))
  server.get(
    '/health',
    asyncHandler(async (req, res) => {
      const args: Args = {
        url: Config.test.url.toString(),
        auth: Config.test.auth,
        viewportWidth: 1500,
        viewportHeight: 1000,
        devicePixelRatio: 4,
      }
      await _generatePDF(args)
      res.send('healthy')
    }),
  )

  server.get(
    '/test',
    asyncHandler(async (req, res) => {
      const args: Args = {
        url: Config.test.url.toString(),
        auth: Config.test.auth,
        viewportWidth: 1500,
        viewportHeight: 1000,
        devicePixelRatio: 4,
      }
      const pdfBuffer = await _generatePDF(args)
      res.set('Content-Type', 'application/pdf')
      res.set('Content-Length', `${pdfBuffer.length}`)
      res.end(pdfBuffer)
    }),
  )
  server.get('/crash', () => {
    throw new Error('crash')
  })

  server.post(
    '/print',
    bodyParser.json(),
    // Note, you won't need async handler in express 5.
    asyncHandler(async (req, res) => {
      if (
        !(req.header('Authorization') ?? '')
          .split(',')
          .some((x) => x.trim() === `Bearer ${Config.interServerToken}`)
      ) {
        res.status(403).send('Forbidden')
        return
      }

      const args = argsGuard(req.body).force()
      const pdfBuffer = await _generatePDF(args)
      res.set('Content-Type', 'application/pdf')
      res.set('Content-Length', `${pdfBuffer.length}`)
      res.end(pdfBuffer)
    }),
  )

  // This should come after all routes.
  server.use(Sentry.Handlers.errorHandler({ shouldHandleError: () => true }))
  server.listen(Config.port)
}
type Args = {
  url: string
  auth: string | null
  viewportWidth: number
  viewportHeight: number
  devicePixelRatio: number
}

const argsGuard: JSONGuard<Args> = object({
  url: string,
  auth: nullable(string),
  viewportWidth: chain(number, integer, gte(1)),
  viewportHeight: chain(number, integer, gte(1)),
  devicePixelRatio: chain(number, integer, gte(1)),
})

const _generatePDF = async (args: Args) => {
  console.dir('generatePDF:start')

  let start = Date.now()
  const sStart = start

  // ---- Create Page ----
  const page = await browser.newPage()
  console.dir(`generatePDF:createPage ${Date.now() - start}`)
  start = Date.now()

  try {
    const result = await Promise.race([
      _pdfFromPage(page, args),
      block(async () => {
        await setTimeout(TIMEOUT)
        return 'timeout' as const
      }),
    ])
    if (result === 'timeout') throw new Error('PDF Generation timeout.')
    console.dir(`generatePDF:total ${Date.now() - sStart}`)
    return result
  } finally {
    await page.close()
  }
}

// Timeouts are needed inside this thought we have timeouts in the
// generatePDF function because some puppeteer timeouts are < TIMEOUT.
const _pdfFromPage = async (page: Page, args: Args) => {
  const { auth, url, viewportWidth, viewportHeight, devicePixelRatio } = args
  let start = Date.now()
  page.on('console', (msg) => {
    console.log('PAGE LOG:', msg.text())
  })
  page.on('pageerror', (error) => {
    console.log('ERROR:')
    console.log(error.message)
    console.log(error.stack)
  })

  // ---- Authenticate ----
  if (auth) {
    const [username, password] = atob(auth).split(':')
    assert(username && password)
    await page.authenticate({ username, password })
    console.dir(`generatePDF:authenticatePage ${Date.now() - start}`)
    start = Date.now()
  }

  // ---- Set page settings ----
  await page.emulateMediaType('print')
  await page.setViewport({
    width: viewportWidth,
    height: viewportHeight,
    // Device scale factor controls the resolution of the canvas elements.
    deviceScaleFactor: devicePixelRatio,
  })
  console.dir(`generatePDF:setPageSettings ${Date.now() - start}`)
  start = Date.now()

  // ---- Goto URL ----
  await page.goto(url, { timeout: TIMEOUT })
  console.dir(`generatePDF:gotoURL ${Date.now() - start}`)
  start = Date.now()

  // ---- Wait for Network ----
  await page.waitForNetworkIdle({ timeout: TIMEOUT })
  console.dir(`generatePDF:waitForNetworkIdle ${Date.now() - start}`)
  start = Date.now()

  // ---- Wait for Ready ----
  await page.waitForFunction('window.__APP_READY_TO_PRINT__', {
    timeout: TIMEOUT,
  })
  console.dir(`generatePDF:waitForReady ${Date.now() - start}`)
  start = Date.now()

  // ---- Wait for Extra 100ms ----
  await setTimeout(100)
  console.dir(`generatePDF:waitForExtra100ms ${Date.now() - start}`)
  start = Date.now()

  // ---- Generate PDF ----
  const pdfBuffer = await page.pdf({
    timeout: TIMEOUT,
    preferCSSPageSize: true,
    printBackground: true,
  })
  console.dir(`generatePDF:pdf ${Date.now() - start}`)

  const isError = await page.evaluate(
    'window.__APP_ERROR__ || !window.__APP_READY_TO_PRINT__',
  )
  if (isError) throw new Error('Trying to generate PDF on error page.')
  return pdfBuffer
}
