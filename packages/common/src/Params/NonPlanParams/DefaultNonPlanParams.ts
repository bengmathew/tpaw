import { NonPlanParams } from './NonPlanParams'

export const getDefaultNonPlanParams = (): NonPlanParams => ({
  v: 21,
  timezone: { type: 'auto' },
  percentileRange: { start: 5, end: 95 },
  numOfSimulationForMonteCarloSampling: 500,
  dev: {
    alwaysShowAllMonths: false,
  },
})

export const defaultNonPlanParams = getDefaultNonPlanParams()
