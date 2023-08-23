import React from 'react'
import { useLazyLoadQuery } from 'react-relay'
import { graphql } from 'relay-runtime'
import { AppPage } from './App/AppPage'
import { RelayTestQuery } from './__generated__/RelayTestQuery.graphql'

// Define a query
const query = graphql`
  query RelayTestQuery {
    ping
  }
`

export const RelayTest = React.memo(() => {
  const data = useLazyLoadQuery<RelayTestQuery>(query, {})

  return (
    <AppPage className="" title="Test" >
      Hello {data.ping}
    </AppPage>
  )
})
