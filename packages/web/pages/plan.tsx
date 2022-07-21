import React from 'react'
import {Plan} from '../src/Pages/Plan/Plan'
import {
  PlanContent,
  planGetStaticProps,
} from '../src/Pages/Plan/PlanGetStaticProps'
import {WithWindowSize} from '../src/Utils/WithWindowSize'

export default React.memo((planContent: PlanContent) => (
  <WithWindowSize>
    <Plan {...planContent} />
  </WithWindowSize>
))

export const getStaticProps = planGetStaticProps
