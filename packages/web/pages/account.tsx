import React from 'react'
import {RequireUser} from '../src/Pages/App/RequireUser'
import {Account} from '../src/Pages/Account/Account'

export default React.memo(() => (
  <RequireUser>
    <Account />
  </RequireUser>
))

