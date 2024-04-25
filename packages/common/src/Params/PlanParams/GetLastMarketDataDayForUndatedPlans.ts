import { getNYZonedTime } from '../../Misc/GetZonedTimeFns'
import { CalendarDay } from './PlanParams'

export const getLastMarketDataDayForUndatedPlans = (
  currentTimestamp: number,
): CalendarDay => {
  const { year, month, day } = getNYZonedTime(currentTimestamp).minus({
    days: 1,
  })
  return { year, month, day }
}
