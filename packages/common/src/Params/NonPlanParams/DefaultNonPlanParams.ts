import { NonPlanParams, currentNonPlanParamsVersion } from './NonPlanParams'

export const getDefaultNonPlanParams = (
  currentTimestamp: number,
): NonPlanParams => ({
  v: currentNonPlanParamsVersion,
  timestamp: currentTimestamp,
  timezone: { type: 'auto' },
  showOfflinePlansMenuSection: false,
  numOfSimulationForMonteCarloSampling: 500,
  pdfReportSettings: {
    pageSize: 'default',
    shouldEmbedLink: 'auto',
  },
  dev: {
    showSyncStatus: false,
    showDevFeatures: false,
    alwaysShowAllMonths: false,
    overridePlanResultChartYRange: false,
  },
})
