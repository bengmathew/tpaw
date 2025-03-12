import * as Sentry from '@sentry/nextjs'
import { API, fGet } from '@tpaw/common'
import _ from 'lodash'
import { DateTime } from 'luxon'
import { toast } from 'react-toastify'
import * as uuid from 'uuid'
import { AppError } from '../Pages/App/AppError'
import { FirebaseUser } from '../Pages/App/WithFirebaseUser'
import { Config } from '../Pages/Config'
import { sendAnalyticsEvent } from './SendAnalyticsEvent'

const sessionId = uuid.v4()

export const fetchGQL =
  (firebaseUser: FirebaseUser | null) =>
  async ({ text }: { text: string | null }, variables: any) => {
    const authHeaders = _.compact([
      Config.client.debug.authHeader,
      firebaseUser ? `Bearer ${await firebaseUser.getIdToken(false)}` : null,
    ])
    const bodyStr = JSON.stringify({ query: text, variables })
    const now = Date.now()
    let response
    try {
      response = await fetch(`${Config.client.urls.backend}/gql`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          ...(authHeaders ? { authorization: authHeaders.join(', ') } : {}),
          'x-iana-timezone-name': fGet(DateTime.local().zoneName),
          'x-app-session-id': sessionId,
          'x-app-api-version': API.version,
          'x-app-client-version': API.clientVersion,
          'x-app-client-timestamp': `${now}`,
        },
        body: bodyStr,
      })
    } catch (e) {
      throw new AppError('networkError')
    }

    if (!response.ok) {
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

    const json = await response.json()
    console.log('json', json)
    if (json.errors) throw new AppError('serverError')
    return json
  }
