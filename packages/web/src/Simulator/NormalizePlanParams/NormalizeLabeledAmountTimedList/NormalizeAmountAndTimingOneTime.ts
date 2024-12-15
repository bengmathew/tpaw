import { LabeledAmountTimed } from '@tpaw/common'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { NormalizedAges, MonthToMFN } from '../NormalizeAges'
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
  monthToMFN: MonthToMFN,
  ages: NormalizedAges,
): NormalizedAmountAndTimingOneTime | null => {
  const asMFN = monthToMFN.pastElided(amountAndTiming.month)
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
