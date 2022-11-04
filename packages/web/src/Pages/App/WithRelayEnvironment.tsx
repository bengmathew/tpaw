import React, { ReactNode, useMemo } from 'react'
import { useFirebaseUser } from './WithFirebaseUser'

import { RelayEnvironmentProvider } from 'react-relay'
import { Environment, Network, RecordSource, Store } from 'relay-runtime'
import { fetchGQL } from '../../Utils/FetchGQL'

export const WithRelayEnvironment = React.memo(
  ({ children }: { children?: ReactNode }) => {
    const firebaseUser = useFirebaseUser()
    const environment = useMemo(
      () =>
        new Environment({
          network: Network.create(fetchGQL(firebaseUser)),
          store: new Store(new RecordSource()),
        }),
      [firebaseUser],
    )

    return (
      <RelayEnvironmentProvider environment={environment}>
        {children}
      </RelayEnvironmentProvider>
    )
  },
)
