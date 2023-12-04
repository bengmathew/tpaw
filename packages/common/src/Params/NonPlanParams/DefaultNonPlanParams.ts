import { NonPlanParams } from './NonPlanParams'

export const getDefaultNonPlanParams = (): NonPlanParams => ({
  v: 23,
  timezone: { type: 'auto' },
  numOfSimulationForMonteCarloSampling: 500,
  dev: {
    showSyncStatus: false,
    showDevFeatures: false,
    alwaysShowAllMonths: false,
    overridePlanResultChartYRange: false,
  },
})

export const defaultNonPlanParams = getDefaultNonPlanParams()
