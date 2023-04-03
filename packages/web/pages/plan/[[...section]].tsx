import React from 'react'
import { WithWindowSize } from '../../src/Pages/App/WithWindowSize'
import { Plan } from '../../src/Pages/Plan/Plan'
import {
  planGetStaticProps,
  PlanStaticProps,
} from '../../src/Pages/Plan/PlanGetStaticProps'
import { paramsInputTypes } from '../../src/Pages/Plan/PlanInput/Helpers/PlanInputType'

export default React.memo(({ content, marketData }: PlanStaticProps) => (
    <Plan planContent={content} marketData={marketData} />
))

export const getStaticProps = planGetStaticProps

export const getStaticPaths = () => ({
  paths: [
    ...paramsInputTypes.map((section) => ({ params: { section: [section] } })),
    { params: { section: ['help'] } },
    { params: { section: null } },
  ],
  fallback: false,
})
