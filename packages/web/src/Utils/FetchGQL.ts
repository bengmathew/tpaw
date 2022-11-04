import _ from 'lodash'
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
      },
      body: JSON.stringify({ query: text, variables }),
    })

    return await response.json()
  }
