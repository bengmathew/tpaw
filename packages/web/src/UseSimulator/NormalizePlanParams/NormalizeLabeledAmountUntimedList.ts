import { LabeledAmountUntimed } from '@tpaw/common'
import _ from 'lodash'

export const normalizeLabeledAmountUntimedList = (
  labeledAmounts: Record<string, LabeledAmountUntimed>,
) => _.values(labeledAmounts).sort((a, b) => a.sortIndex - b.sortIndex)
