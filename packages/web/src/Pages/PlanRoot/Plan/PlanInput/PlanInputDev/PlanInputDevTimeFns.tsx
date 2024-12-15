import { useDailyMarketSeriesSrc, useSimulationInfo } from '../../../PlanRootHelpers/WithSimulation'


export const useIsPlanInputDevTimeFastForwardCardModified = () => {
  const { fastForwardInfo } = useSimulationInfo()
  return fastForwardInfo.isFastForwarding
}

export const useIsPlanInputDevTimeSynthesizeMarketDataCardModified = () => {
  const { dailyMarketSeriesSrc } = useDailyMarketSeriesSrc()
  return dailyMarketSeriesSrc.type !== 'live'
}

export const useIsPlanInputDevTimeModified = () => {
  const m1 = useIsPlanInputDevTimeFastForwardCardModified()
  const m2 = useIsPlanInputDevTimeSynthesizeMarketDataCardModified()
  return m1 || m2
}