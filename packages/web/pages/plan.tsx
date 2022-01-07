import React from 'react'
import { Plan } from '../src/Pages/Plan/Plan'
import { WithWindowSize } from '../src/Utils/WithWindowSize'

export default React.memo(() => (
  <WithWindowSize>
    <Plan />
  </WithWindowSize>
))
