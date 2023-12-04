import * as Sentry from '@sentry/nextjs'
import { API, fGet } from '@tpaw/common'
import _ from 'lodash'
import { DateTime } from 'luxon'
import { toast } from 'react-toastify'
import * as uuid from 'uuid'
import { AppError } from '../Pages/App/AppError'
import { FirebaseUser } from '../Pages/App/WithFirebaseUser'
import { Config } from '../Pages/Config'

const sessionId = uuid.v4()

export const fetchGQL =
  (firebaseUser: FirebaseUser | null) =>
  async ({ text }: { text: string | null }, variables: any) => {
    const authHeaders = _.compact([
      Config.client.debug.authHeader,
      firebaseUser ? `Bearer ${await firebaseUser.getIdToken(false)}` : null,
    ])
    const bodyStr = JSON.stringify({ query: text, variables })
    const response = await fetch(`${Config.client.urls.backend}/gql`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...(authHeaders ? { authorization: authHeaders.join(', ') } : {}),
        'x-iana-timezone-name': fGet(DateTime.local().zoneName),
        'x-app-session-id': sessionId,
        'x-app-api-version': API.version,
        'x-app-client-version': API.clientVersion,
      },
      body: bodyStr,
    })

    if (!response.ok) {
      if (response.status === 413) {
        Sentry.captureMessage(`413 error. bodyStr.length: ${bodyStr.length}`)
      }
      const code = response.headers.get('x-app-error-code')
      switch (code) {
        case 'downForMaintenance':
          throw new AppError('serverDownForMaintenance')
        case 'downForUpdate':
          throw new AppError('serverDownForUpdate')
        case 'clientNeedsUpdate':
          throw new AppError('clientNeedsUpdate')
      }
      throw new AppError('networkError')
    }
    if (response.headers.get('x-app-new-client-version') === 'true') {
      toast(
        'A new version of the planner is available. Please reload to get the lateset version.',
        {
          type: 'info',
          toastId: 'newClientVersion',
          autoClose: false,
        },
      )
    }
    return await response.json()
  }
