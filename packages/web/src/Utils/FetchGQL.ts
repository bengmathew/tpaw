import { API, fGet } from '@tpaw/common'
import _ from 'lodash'
import { DateTime } from 'luxon'
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
    const response = await fetch(`${Config.client.urls.backend}/gql`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...(authHeaders ? { authorization: authHeaders.join(', ') } : {}),
        'x-iana-timezone-name': fGet(DateTime.local().zoneName),
        'x-app-session-id': sessionId,
        'x-app-api-version': API.version,
      },
      body: JSON.stringify({ query: text, variables }),
    })
    const code = response.headers.get('x-app-error-code')
    console.dir(code)
    switch (code) {
      case 'downForMaintenance':
        throw new AppError('serverDownForMaintenance')
      case 'downForUpdate':
        throw new AppError('serverDownForUpdate')
      case 'clientNeedsUpdate':
        throw new AppError('clientNeedsUpdate')
    }
    return await response.json()
  }
