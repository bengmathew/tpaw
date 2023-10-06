import { getStatsWithLog, sequentialAnnualReturnsFromMonthly } from '../Utils'
import { returnsFromRawInput } from './HistoricalReturnsHelpers'
import { rawHistoricalMonthlyDataAsCSV } from './RawHistoricalMonthlyDataAsCSV'

// const annual = returnsFromRawInput(rawHistoricalAnnualDataAsCSV)
const monthly = returnsFromRawInput(rawHistoricalMonthlyDataAsCSV)
export const historicalReturns = {
  // annual,
  monthly: {
    ...monthly,
    annualStats: {
      stocks: getStatsWithLog(
        sequentialAnnualReturnsFromMonthly(monthly.stocks.ofBase.returns),
      ),
      bonds: getStatsWithLog(
        sequentialAnnualReturnsFromMonthly(monthly.bonds.ofBase.returns),
      ),
    },
  },
}
