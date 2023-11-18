import { NonPlanParams } from './NonPlanParams'

export const getDefaultNonPlanParams = (): NonPlanParams => ({
  v: 22,
  timezone: { type: 'auto' },
  numOfSimulationForMonteCarloSampling: 500,
  dev: {
    showDevFeatures: false,
    alwaysShowAllMonths: false,
    overridePlanResultChartYRange: false,
  },
})

export const defaultNonPlanParams = getDefaultNonPlanParams()
