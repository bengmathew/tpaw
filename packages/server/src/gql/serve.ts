import { ApolloServer, ApolloServerPlugin } from '@apollo/server'
import { expressMiddleware } from '@apollo/server/express4'
import Sentry from '@sentry/node'
import Tracing from '@sentry/tracing'
import {
  assert,
  fGet,
  getDefaultNonPlanParams,
  getDefaultPlanParams,
  getSlug,
} from '@tpaw/common'
import { ApolloError } from 'apollo-server-express'
import bodyParser from 'body-parser'
import compression from 'compression'
import cors from 'cors'
import express from 'express'
import { writeFileSync } from 'fs'
import { lexicographicSortSchema, printSchema } from 'graphql'
import path from 'path'
import * as uuid from 'uuid'
import { cli } from '../CLI/CLI.js'
import { pushMarketData } from '../CLI/CLIMisc/CLIMiscPushMarketData.js'
import { Clients } from '../Clients.js'
import { Config } from '../Config.js'
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
  // GCP Cloud Run does not gzip for us so do it here.
  server.use(compression())
  server.get('/', (req, res) => res.send('I am root!'))
  server.get('/ping', (req, res) => res.send('pong'))
  server.get('/marketDataURL', async (req, res) => {
    const bucket = Clients.gcs.bucket(Config.google.marketDataBucket)
    const [currentLatest] = await bucket.getFiles({ prefix: 'latest/' })
    assert(currentLatest.length === 1)
    const file = fGet(currentLatest[0])
    const [fileName] = await file.getSignedUrl({
      action: 'read',
      expires: Date.now() + 1000 * 60 * 60 * 24 * 7,
    })
    res.send(fileName)
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

  const apollo = new ApolloServer<Context>({ schema, plugins: [sentryPlugin] })
  await apollo.start()
  server.use(
    '/gql',
    cors({ origin: Config.websiteURL }),
    bodyParser.json(),
    expressMiddleware<Context>(apollo, {
      context: async ({ req }) => {
        try {
          const ianaTimezoneName =
            req.headers['X-IANA-Timezone-Name'.toLowerCase()]
          assert(typeof ianaTimezoneName === 'string')

          const user = await (async () => {
            const idToken = (req.headers.authorization || '')
              .split(', ')
              .filter((x) => x.startsWith('Bearer'))
              .map((x) => x.substring('Bearer '.length))[0]

            if (!idToken) return null
            const decodedToken =
              await Clients.firebaseAuth.verifyIdToken(idToken)
            const userId = decodedToken.uid

            const label = null
            const now = new Date()
            const planParams = getDefaultPlanParams(
              now.getTime(),
              ianaTimezoneName,
            )
            await Clients.prisma.user.upsert({
              create: {
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
              },
              update: {
                clientIANATimezoneName: ianaTimezoneName,
              },
              where: { id: userId },
            })
            return { id: decodedToken.uid }
          })()
          return { user }
        } catch (e) {
          console.dir(e)
          Sentry.captureException(e)
          throw e
        }
      },
    }),
  )

  server.listen(Config.port)
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
          if (err instanceof ApolloError) {
            continue
          }
          // Add scoped report details and send to Sentry
          Sentry.withScope((scope) => {
            // Annotate whether failing operation was query/mutation/subscription
            scope.setTag('kind', ctx.operation?.operation)
            if (err.path) {
              // We can also add the path as breadcrumb
              scope.addBreadcrumb({
                category: 'query-path',
                message: err.path.join(' > '),
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
