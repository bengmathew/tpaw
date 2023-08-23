import { API } from '@tpaw/common'
import { JSONGuard } from 'json-guard'
import { Clients } from '../../../Clients.js'
import { Config } from '../../../Config.js'
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
      const url = new URL(`${Config.websiteURL}/auth/email`)
      url.searchParams.set('dest', dest)
      const link = await Clients.firebaseAuth.generateSignInWithEmailLink(
        email,
        {
          url: url.toString(),
          handleCodeInApp: true,
        },
      )
      await Clients.postmark.sendEmailWithTemplate({
        TemplateAlias: 'login-link',
        From: 'hello@tpawplanner.com',
        To: email,
        TemplateModel: {
          name: email.split('@')[0],
          action_url: link,
        },
      })
      return success
    },
  }),
)
