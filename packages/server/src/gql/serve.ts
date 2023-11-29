import { ApolloServer, ApolloServerPlugin } from '@apollo/server'
import { expressMiddleware } from '@apollo/server/express4'
import { Prisma } from '@prisma/client'
import Sentry from '@sentry/node'
import Tracing from '@sentry/tracing'
import {
  API,
  assert,
  block,
  fGet,
  getDefaultNonPlanParams,
  getDefaultPlanParams,
  getSlug,
} from '@tpaw/common'
import bodyParser from 'body-parser'
import chalk from 'chalk'
import compression from 'compression'
import cors from 'cors'
import express from 'express'
import { writeFileSync } from 'fs'
import { lexicographicSortSchema, printSchema } from 'graphql'
import { DateTime } from 'luxon'
import morgan from 'morgan'
import path from 'path'
import * as uuid from 'uuid'
import { cli } from '../CLI/CLI.js'
import { pushMarketData } from '../CLI/CLIMisc/CLIMiscPushMarketData.js'
import { Clients } from '../Clients.js'
import { Config } from '../Config.js'
import { serialTransaction } from '../Utils/PrismaTransaction.js'
import { Context } from './Context.js'
import { patchPlanParams } from './GQLUser/GQLUserPlan/PatchPlanParams.js'
import { schema } from './schema.js'

cli.command('serve').action(async () => await _impl())

async function _impl() {
  if (process.env['NODE_ENV'] !== 'production') {
    writeFileSync(
      path.join(process.cwd(), '/generated/schema.graphql'),
      printSchema(lexicographicSortSchema(schema)),
    )
  }
  const server = express()
  Sentry.init({
    dsn: Config.sentry.dsn,
    integrations: [
      // DO NOT enable HTTP calls tracing. This uses up quota pretty fast.
      // new Sentry.Integrations.Http({tracing: true}),
      // enable Express.js middleware tracing
      new Tracing.Integrations.Express({ app: server }),
    ],

    // We recommend adjusting this value in production, or using tracesSampler
    // for finer control
    tracesSampleRate: 1.0,
  })
  server.use(Sentry.Handlers.errorHandler({ shouldHandleError: () => true }))
  if (!Config.isProduction) server.use(morgan('tiny'))

  server.use(cors({ origin: Config.websiteURL }))
  // GCP Cloud Run does not gzip for us so do it here.
  server.use(compression())
  server.use((req, res, next) => {
    if (Config.status.downForMaintenance || Config.status.downForUpdate) {
      const code = Config.status.downForMaintenance
        ? 'downForMaintenance'
        : 'downForUpdate'
      res.status(503)
      res.setHeader('Access-Control-Expose-Headers', 'x-app-error-code')
      res.setHeader('x-app-error-code', code)
      res.send(code)
    } else {
      next()
    }
  })

  server.get('/', (req, res) => res.send('I am root!'))
  server.get('/ping', (req, res) => res.send('pong'))
  server.get('/time', (req, res) => res.send(`${Date.now()}`))
  server.get('/crash', (req, res) => {
    throw new Error('crash')
  })
  
  server.get('/marketDataURL', async (req, res) => {
    const bucket = Clients.gcs.bucket(Config.google.marketDataBucket)
    const [currentLatest] = await bucket.getFiles({ prefix: 'latest/' })
    assert(currentLatest.length === 1)
    const file = fGet(currentLatest[0])
    const filename = await file.publicUrl()
    res.send(filename)
  })

  server.get('/deploy-frontend', async (req, res) => {
    if (req.query['token'] !== Config.deployFrontEnd.token) {
      res.status(401)
      res.send('Unauthorized')
    } else {
      await pushMarketData()
      await fetch(Config.deployFrontEnd.url)
      res.send('ok')
    }
  })
  const apollo = new ApolloServer<Context>({
    schema,
    plugins: [loggingPlugin, sentryPlugin],
  })
  await apollo.start()
  server.use('/gql', (req, res, next) => {
    const apiVersion = req.headers['x-app-api-version']
    if (apiVersion !== API.version) {
      res.status(400)
      res.setHeader('Access-Control-Expose-Headers', 'x-app-error-code')
      res.setHeader('x-app-error-code', 'clientNeedsUpdate')
      res.send('clientNeedsUpdate')
    } else {
      next()
    }
  })
  server.use(
    '/gql',
    bodyParser.json({limit:'10mb'}),
    expressMiddleware<Context>(apollo, {
      context: async ({ req }) => {
        try {
          const apiVersion = req.headers['x-app-api-version']
          if (apiVersion !== API.version) {
          }
          const ianaTimezoneName = req.headers['x-iana-timezone-name']
          assert(typeof ianaTimezoneName === 'string')
          const sessionId = req.headers['x-app-session-id'] ?? null
          assert(sessionId === null || typeof sessionId === 'string')

          return await block<Promise<Context>>(async () => {
            const idToken = (req.headers.authorization || '')
              .split(', ')
              .filter((x) => x.startsWith('Bearer'))
              .map((x) => x.substring('Bearer '.length))[0]

            if (!idToken) return { user: null, sessionId }
            const decodedToken =
              await Clients.firebaseAuth.verifyIdToken(idToken)
            const userId = decodedToken.uid

            const label = null
            const now = new Date()
            const planParams = getDefaultPlanParams(
              now.getTime(),
              ianaTimezoneName,
            )
            const createData: Prisma.UserCreateInput = {
              id: userId,
              planWithHistory: {
                create: {
                  planId: uuid.v4(),
                  isMain: true,
                  label,
                  slug: getSlug(label, []),
                  addedToServerAt: now,
                  sortTime: now,
                  lastSyncAt: now,
                  resetCount: 0,
                  endingParams: planParams,
                  paramsChangeHistory: {
                    create: patchPlanParams.generate({ type: 'forCreate' }, [
                      {
                        id: uuid.v4(),
                        params: planParams,
                        change: { type: 'start', value: null },
                      },
                    ]),
                  },
                  reverseHeadIndex: 0,
                },
              },
              nonPlanParamsLastUpdatedAt: now,
              nonPlanParams: getDefaultNonPlanParams(),
              clientIANATimezoneName: ianaTimezoneName,
            }

            const updateData: Prisma.UserUpdateInput = {
              clientIANATimezoneName: ianaTimezoneName,
            }

            await serialTransaction(
              async (tx) => {
                const user = await tx.user.findUnique({
                  where: { id: userId },
                })
                if (user) {
                  await tx.user.update({
                    where: { id: userId },
                    data: updateData,
                  })
                } else {
                  await tx.user.create({ data: createData })
                }
              },
              {
                retries: 3,
                onConflict: () => {
                  const time = Date.now()
                  const timeStr = DateTime.fromMillis(time, {
                    zone: 'America/Los_Angeles',
                  }).toLocaleString(DateTime.DATETIME_FULL_WITH_SECONDS)
                  const queryStr = (req.body.query as string) ?? ''
                  const queryStrLine1 = queryStr.slice(
                    0,
                    queryStr.indexOf('\n'),
                  )
                  const text = `Transaction conflict on context creation\nuser: ${userId}\ntime: ${time}\ntimeStr: ${timeStr}\nsessionId: ${sessionId}\nquery: \n${queryStrLine1})}`
                  if (Config.isProduction) {
                    Sentry.captureMessage(text)
                  } else {
                    console.log(text)
                  }
                },
              },
            )
            return {
              user: { id: userId },
              sessionId: sessionId,
            }
          })
        } catch (e) {
          if (!Config.isProduction) console.dir(e)
          Sentry.captureException(e)
          throw e
        }
      },
    }),
  )

  server.listen(Config.port)
}

// From: https://www.apollographql.com/docs/apollo-server/monitoring/metrics/#logging
const loggingPlugin: ApolloServerPlugin = {
  // Fires whenever a GraphQL request is received from a client.

  async requestDidStart(requestContext) {
    const startTime = Date.now()

    return {
      async willSendResponse(requestContext) {
        const { sessionId } = requestContext.contextValue as Context
        const durationStr = `${(Date.now() - startTime).toFixed(0)}ms`
        const statusStr = `${requestContext.response.http.status ?? '200'}`
        const timeStr = DateTime.fromMillis(startTime, {
          zone: 'America/Los_Angeles',
        }).toLocaleString(DateTime.DATETIME_FULL_WITH_SECONDS)
        if (Config.isProduction) {
          const log = Clients.logging.log('gql')
          void log.write(
            log.entry(
              { resource: { type: 'global' }, severity: 'INFO' },
              {
                time: startTime,
                timeStr,
                statusStr,
                sessionId,
                duration: durationStr,
                operationName: requestContext.operationName,
              },
            ),
          )
        } else {
          console.log(
            chalk.dim(
              `    ${statusStr.padStart(5)} - ${durationStr.padStart(
                8,
              )} - ${timeStr} - ${requestContext.operationName ?? '-'}`,
            ),
          )
        }
      },
    }
  },
}

// From: https://blog.sentry.io/2020/07/22/handling-graphql-errors-using-sentry
const sentryPlugin: ApolloServerPlugin = {
  async requestDidStart() {
    return {
      async didEncounterErrors(ctx) {
        // If we couldn't parse the operation, don't
        // do anything here
        if (!ctx.operation) {
          return
        }
        for (const err of ctx.errors) {
          // Only report internal server errors,
          // all errors extending ApolloError should be user-facing
          // (NOTE: deviation from sentry doc. ApolloError was removed in Apollo 4, and replaced with GraphQLError).
          // see: https://www.apollographql.com/docs/apollo-server/migration/#apolloerror
          // (NOTE2: Not filtering these errors.)
          // if (err instanceof GraphQLError) {
          //   continue
          // }

          // Add scoped report details and send to Sentry
          Sentry.withScope((scope) => {
            const { user, sessionId } = ctx.contextValue as Context
            scope.setUser(user ? { id: user.id } : null)
            // Annotate whether failing operation was query/mutation/subscription
            scope.setTag('kind', ctx.operation?.operation)
            // Log query and variables as extras
            // (make sure to strip out sensitive data!)
            scope.setExtra('sessionId', sessionId)
            scope.setExtra('query', ctx.request.query)
            scope.setExtra('variables', ctx.request.variables)
            if ((err as any).path) {
              // We can also add the path as breadcrumb
              scope.addBreadcrumb({
                category: 'query-path',
                message: (err as any).path.join(' > '),
                level: 'debug',
              })
            }
            console.dir(err)
            Sentry.captureException(err)
          })
        }
      },
    }
  },
}
