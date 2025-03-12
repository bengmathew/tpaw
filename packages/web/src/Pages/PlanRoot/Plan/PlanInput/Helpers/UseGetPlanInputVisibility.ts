import {
  DialogPosition,
  PLAN_PARAMS_CONSTANTS,
  block,
  noCase,
} from '@tpaw/common'
import { PlanParamsNormalized } from '@tpaw/common'
import { PlanInputType } from './PlanInputType'
import { isPlanSectionDialogInOverlayMode } from './PlanSectionDialogPosition'

export const useGetPlanInputVisibility = (
  planParamsNorm: PlanParamsNormalized,
) => {
  const { dialogPosition } = planParamsNorm
  const isFutureSavingsAllowed =
    planParamsNorm.ages.validMonthRangesAsMFN.futureSavings !== null

  const _helper = (
    showAtDialogPosition: Exclude<
      DialogPosition,
      'show-results' | 'show-all-inputs'
    >,
    visible = true,
  ) => {
    if (!visible) return { visible: false } as const

    const disabled = block(() => {
      if (isPlanSectionDialogInOverlayMode(dialogPosition.effective))
        return true
      return (
        PLAN_PARAMS_CONSTANTS.dialogPositionOrder.indexOf(
          dialogPosition.effective,
        ) <
        PLAN_PARAMS_CONSTANTS.dialogPositionOrder.indexOf(showAtDialogPosition)
      )
    })
    return {
      visible: true,
      disabled,
      // In overlay mode, the curtain will effectively do the lightening, so don't
      // lighten the button directly.
      grayOutButton:
        disabled && !isPlanSectionDialogInOverlayMode(dialogPosition.effective),
    }
  }

  return (type: PlanInputType) => {
    switch (type) {
      case 'age':
        return _helper('age')
      case 'current-portfolio-balance':
        return _helper('current-portfolio-balance')
      case 'future-savings':
        return _helper('future-savings', isFutureSavingsAllowed)
      case 'income-during-retirement':
        return _helper('income-during-retirement')
      case 'extra-spending':
        return _helper('done')
      case 'legacy':
        return _helper('done', planParamsNorm.advanced.strategy !== 'SWR')
      case 'spending-ceiling-and-floor':
        return _helper('done', planParamsNorm.advanced.strategy !== 'SWR')
      case 'risk':
        return _helper('done')
      case 'expected-returns-and-volatility':
        return _helper('done')
      case 'inflation':
        return _helper('done')
      case 'simulation':
        return _helper('done')
      case 'strategy':
        return _helper('done')
      case 'dev-misc':
        return _helper('done')
      case 'dev-simulations':
        return _helper('done')
      case 'dev-time':
        return _helper('done')
      default:
        noCase(type)
    }
  }
}
