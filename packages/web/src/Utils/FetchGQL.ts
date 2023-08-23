import _ from 'lodash'
import { DateTime } from 'luxon'
import { FirebaseUser } from '../Pages/App/WithFirebaseUser'
import { Config } from '../Pages/Config'
import { fGet } from '@tpaw/common'

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
    return await response.json()
  }
