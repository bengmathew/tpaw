import _ from 'lodash'
import { LabeledAmountUntimed } from '../PlanParams'

export const normalizeLabeledAmountUntimedList = (
  labeledAmounts: Record<string, LabeledAmountUntimed>,
) => _.values(labeledAmounts).sort((a, b) => a.sortIndex - b.sortIndex)
