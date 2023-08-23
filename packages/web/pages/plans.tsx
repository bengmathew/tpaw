import React from 'react'
import { RequireUser } from '../src/Pages/App/RequireUser'
import { Plans } from '../src/Pages/Plans/Plans'

export default React.memo(() => (
  <RequireUser>
    <Plans />
  </RequireUser>
))
