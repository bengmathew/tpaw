import React from 'react'
import {Plan, PlanContent, planGetStaticProps} from '../src/Pages/Plan/Plan'
import {WithWindowSize} from '../src/Utils/WithWindowSize'

export default React.memo((planContent: PlanContent) => (
  <WithWindowSize>
    <Plan {...planContent} />
  </WithWindowSize>
))

export const getStaticProps = planGetStaticProps
