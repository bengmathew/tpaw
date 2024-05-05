import {
  CalendarMonth,
  LabeledAmountTimed,
  LabeledAmountTimedList,
  block,
  noCase,
} from '@tpaw/common'
import _, { now } from 'lodash'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { NormalizedAges, MonthToMFN } from '../NormalizeAges'
import {
  NormalizedAmountAndTimingOneTime,
  normalizeAmountAndTimingOneTime,
} from './NormalizeAmountAndTimingOneTime'
import {
  NormalizedAmountAndTimingRecurring,
  normalizeAmountAndTimingRecurring,
} from './NormalizeAmountAndTimingRecurring'

export type NormalizedLabeledAmountTimed = {
  label: string | null
  nominal: boolean
  id: string
  sortIndex: number
  colorIndex: number
  amountAndTiming:
    | { type: 'inThePast' }
    | NormalizedAmountAndTimingOneTime
    | NormalizedAmountAndTimingRecurring
}

export const normalizeLabeledAmountTimedList = (
  x: LabeledAmountTimedList,
  validRangeAsMFN: SimpleRange | null,
  monthToMFN: MonthToMFN,
  ages: NormalizedAges,
  nowAsCalendarDay: CalendarMonth | null,
): NormalizedLabeledAmountTimed[] =>
  !validRangeAsMFN
    ? []
    : _.values(x)
        .map((x) =>
          _normalizeLabeledAmountTimed(
            x,
            validRangeAsMFN,
            monthToMFN,
            ages,
            nowAsCalendarDay,
          ),
        )
        .sort((a, b) => a.sortIndex - b.sortIndex)

const _normalizeLabeledAmountTimed = (
  x: LabeledAmountTimed,
  validRangeAsMFN: SimpleRange,
  monthToMFN: MonthToMFN,
  ages: NormalizedAges,
  nowAsCalendarDay: CalendarMonth | null,
): NormalizedLabeledAmountTimed => {
  const amountAndTiming = block(
    (): NormalizedLabeledAmountTimed['amountAndTiming'] | null => {
      switch (x.amountAndTiming.type) {
        case 'oneTime': {
          return normalizeAmountAndTimingOneTime(
            x.amountAndTiming,
            validRangeAsMFN,
            monthToMFN,
            ages,
          )
        }
        case 'recurring': {
          return normalizeAmountAndTimingRecurring(
            x.amountAndTiming,
            validRangeAsMFN,
            monthToMFN,
            ages,
            nowAsCalendarDay,
          )
        }
        case 'inThePast':
          return { type: 'inThePast' }
        default:
          noCase(x.amountAndTiming)
      }
    },
  ) ?? { type: 'inThePast' }
  return {
    label: x.label,
    nominal: x.nominal,
    id: x.id,
    sortIndex: x.sortIndex,
    colorIndex: x.colorIndex,
    amountAndTiming,
  }
}
