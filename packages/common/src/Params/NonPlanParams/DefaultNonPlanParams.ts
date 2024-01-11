import { NonPlanParams } from './NonPlanParams'

export const getDefaultNonPlanParams = (
  currentTimestamp: number,
): NonPlanParams => ({
  v: 24,
  timestamp: currentTimestamp,
  timezone: { type: 'auto' },
  numOfSimulationForMonteCarloSampling: 500,
  pdfReportSettings: {
    pageSize: 'default',
    embeddedLinkType: 'default',
  },
  dev: {
    showSyncStatus: false,
    showDevFeatures: false,
    alwaysShowAllMonths: false,
    overridePlanResultChartYRange: false,
  },
})

