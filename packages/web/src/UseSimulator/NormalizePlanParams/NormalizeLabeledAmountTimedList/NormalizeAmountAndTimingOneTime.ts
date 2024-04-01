import { LabeledAmountTimed } from '@tpaw/common'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { NormalizedAges, ToMFN } from '../NormalizePlanParamsAges'
import {
  NormalizedMonthNotInThePast,
  getNormalizedMonthNotInThePast,
} from './NormalizedMonth'

export type NormalizedAmountAndTimingOneTime = {
  type: 'oneTime'
  amount: number
  month: NormalizedMonthNotInThePast
}

export const normalizeAmountAndTimingOneTime = (
  amountAndTiming: Extract<
    LabeledAmountTimed['amountAndTiming'],
    { type: 'oneTime' }
  >,
  validRangeAsMFN: SimpleRange,
  toMFN: ToMFN,
  ages: NormalizedAges,
): NormalizedAmountAndTimingOneTime | null => {
  const asMFN = toMFN.forMonth.pastElided(amountAndTiming.month)
  return asMFN === 'inThePast'
    ? null
    : {
        type: 'oneTime',
        amount: amountAndTiming.amount,
        month: getNormalizedMonthNotInThePast(
          asMFN,
          amountAndTiming.month,
          {
            includingLocalConstraints: validRangeAsMFN,
            excludingLocalConstraints: validRangeAsMFN,
          },
          { type: 'rangeCheckAndSquishRangeForAge', ages },
        ),
      }
}
