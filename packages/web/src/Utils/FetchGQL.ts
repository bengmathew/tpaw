import { fGet } from '@tpaw/common'
import _ from 'lodash'
import { DateTime } from 'luxon'
import { AppError } from '../Pages/App/AppError'
import { FirebaseUser } from '../Pages/App/WithFirebaseUser'
import { Config } from '../Pages/Config'

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
        'Content-Type': 'application/json',
        ...(authHeaders ? { Authorization: authHeaders.join(', ') } : {}),
        'X-IANA-Timezone-Name': fGet(DateTime.local().zoneName),
      },
      body: JSON.stringify({ query: text, variables }),
    })
    if (response.status === 503) {
      throw new AppError('serverDownForMaintenance')
    }
    return await response.json()
  }
