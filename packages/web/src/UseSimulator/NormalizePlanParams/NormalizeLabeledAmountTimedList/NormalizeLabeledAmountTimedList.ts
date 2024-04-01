import {
  LabeledAmountTimed,
  LabeledAmountTimedList,
  block,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { NormalizedAges, ToMFN } from '../NormalizePlanParamsAges'
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
  toMFN: ToMFN,
  ages: NormalizedAges,
): NormalizedLabeledAmountTimed[] =>
  !validRangeAsMFN
    ? []
    : _.values(x)
        .map((x) => _normalizeLabeledAmountTimed(x, validRangeAsMFN, toMFN, ages))
        .sort((a, b) => a.sortIndex - b.sortIndex)

const _normalizeLabeledAmountTimed = (
  x: LabeledAmountTimed,
  validRangeAsMFN: SimpleRange,
  toMFN: ToMFN,
  ages: NormalizedAges,
): NormalizedLabeledAmountTimed => {
  const amountAndTiming = block(
    (): NormalizedLabeledAmountTimed['amountAndTiming'] | null => {
      switch (x.amountAndTiming.type) {
        case 'oneTime': {
          return normalizeAmountAndTimingOneTime(
            x.amountAndTiming,
            validRangeAsMFN,
            toMFN,
            ages,
          )
        }
        case 'recurring': {
          return normalizeAmountAndTimingRecurring(
            x.amountAndTiming,
            validRangeAsMFN,
            toMFN,
            ages
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
