import { ApolloServer, ApolloServerPlugin } from '@apollo/server'
import { expressMiddleware } from '@apollo/server/express4'
import { Prisma } from '@prisma/client'
import Sentry from '@sentry/node'
import Tracing from '@sentry/tracing'
import { ApolloError } from 'apollo-server-express'
import bodyParser from 'body-parser'
import compression from 'compression'
import cors from 'cors'
import express from 'express'
import { writeFileSync } from 'fs'
import { lexicographicSortSchema, printSchema } from 'graphql'
import path from 'path'
import { cli } from '../CLI/CLI.js'
import { Clients } from '../Clients.js'
import { Config } from '../Config.js'
import { Context } from './Context.js'
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

  const apollo = new ApolloServer<Context>({ schema, plugins: [sentryPlugin] })
  await apollo.start()
  server.use(
    '/gql',
    cors({ origin: Config.websiteURL }),
    bodyParser.json(),
    expressMiddleware<Context>(apollo, {
      context: async ({ req }) => {
        const user = await (async () => {
          const idToken = (req.headers.authorization || '')
            .split(', ')
            .filter((x) => x.startsWith('Bearer'))
            .map((x) => x.substring('Bearer '.length))[0]

          if (!idToken) return null
          const decodedToken = await Clients.firebaseAuth.verifyIdToken(idToken)
          const user: Prisma.UserCreateInput & Prisma.UserUpdateInput = {
            id: decodedToken.uid,
          }
          await Clients.prisma.user.upsert({
            create: user,
            update: {},
            where: { id: user.id },
          })
          return { id: decodedToken.uid }
        })()
        return { user }
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
