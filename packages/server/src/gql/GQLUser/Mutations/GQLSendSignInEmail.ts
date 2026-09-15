import { API } from '@tpaw/common'
import { JSONGuard } from 'json-guard'
import { randomUUID } from 'node:crypto'
import { Clients } from '../../../Clients.js'
import { Config } from '../../../Config.js'
import {
  describeEmailError,
  postmarkDiagnosticLog,
} from '../../../Utils/PostmarkDiagnostics.js'
import { success } from '../../../impl/Common/Success.js'
import { builder } from '../../builder.js'

const Input = builder.inputType('SendSignInEmailInput', {
  fields: (t) => ({ email: t.string(), dest: t.string() }),
})

builder.mutationField('sendSignInEmail', (t) =>
  t.field({
    type: 'Success',
    args: {
      input: t.arg({ type: Input }),
    },
    resolve: async (_, { input }) => {
      const guard: JSONGuard<typeof input> = API.SendSignInEmail.check
      const { email, dest } = guard(input).force()
      const attemptId = randomUUID()
      const startedAt = Date.now()
      const log = (
        event: string,
        details: Record<string, unknown> = {},
        severity: 'INFO' | 'ERROR' = 'INFO',
      ) => {
        console.log(
          JSON.stringify({
            severity,
            message: `sendSignInEmail: ${event}`,
            operation: 'sendSignInEmail',
            attemptId,
            event,
            elapsedMs: Date.now() - startedAt,
            ...details,
          }),
        )
      }
      return postmarkDiagnosticLog.run(log, async () => {
        let stage = 'firebase.generateLink'
        try {
          const url = Config.frontend.paths.auth.email()
          url.searchParams.set('dest', dest)
          log('firebase.started', {
            frontendOrigin: url.origin,
            recipientDomain: email.split('@').pop(),
            nodeVersion: process.version,
          })
          const link = await Clients.firebaseAuth.generateSignInWithEmailLink(
            email,
            {
              url: url.toString(),
              handleCodeInApp: true,
            },
          )
          log('firebase.succeeded')

          stage = 'postmark.sendEmail'
          const client = Clients.postmark
          log('postmark.started', {
            templateAlias: 'login-link',
            from: 'hello@tpawplanner.com',
            requestHost: client.getClientOptions().requestHost,
            timeoutSeconds: client.getClientOptions().timeout,
            postmarkVersion: client.clientVersion,
          })
          const result = await client.sendEmailWithTemplate({
            TemplateAlias: 'login-link',
            From: 'hello@tpawplanner.com',
            To: email,
            TemplateModel: {
              name: email.split('@')[0],
              action_url: link,
            },
          })
          log('postmark.succeeded', {
            postmarkMessageId: result.MessageID,
            postmarkErrorCode: result.ErrorCode,
          })
          return success
        } catch (error) {
          log('failed', { stage, error: describeEmailError(error) }, 'ERROR')
          throw error
        }
      })
    },
  }),
)
