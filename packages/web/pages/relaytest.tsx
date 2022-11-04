import React from 'react'
import { RequireUser } from '../src/Pages/App/RequireUser'
import { RelayTest } from '../src/Pages/RelayTest'

export default React.memo(() => (
  <RequireUser>
    <RelayTest />
  </RequireUser>
))
