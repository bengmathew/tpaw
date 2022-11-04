import React from 'react'
import {RequireUser} from '../src/Pages/App/RequireUser'
import {Login} from '../src/Pages/Login'

export default React.memo(() => (
  <RequireUser>
    <Login />
  </RequireUser>
))

