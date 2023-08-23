import { PlanParams, block, noCase } from '@tpaw/common'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputType } from './PlanInputType'
import {
  isPlanSectionDialogInOverlayMode,
  planSectionDialogOrder,
} from './PlanSectionDialogPosition'

export const useGetPlanInputVisibility = () => {
  const { planParams, planParamsExt } = useSimulation()
  const { isFutureSavingsAllowed } = planParamsExt

  const _helper = (
    showAtDialogPosition: Exclude<
      PlanParams['dialogPosition'],
      'show-results' | 'show-all-inputs'
    >,
    visible = true,
  ) => {
    if (!visible) return { visible: false } as const

    const disabled = block(() => {
      if (isPlanSectionDialogInOverlayMode(planParams.dialogPosition))
        return true
      return (
        planSectionDialogOrder.indexOf(planParams.dialogPosition) <
        planSectionDialogOrder.indexOf(showAtDialogPosition)
      )
    })
    return {
      visible: true,
      disabled,
      // In overlay mode, the curtain will effectively do the lightening, so don't
      // lighten the button directly.
      grayOutButton:
        disabled &&
        !isPlanSectionDialogInOverlayMode(planParams.dialogPosition),
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
        return _helper('done', planParams.advanced.strategy !== 'SWR')
      case 'spending-ceiling-and-floor':
        return _helper('done', planParams.advanced.strategy !== 'SWR')
      case 'risk':
        return _helper('done')
      case 'expected-returns':
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
      case 'dev-historical-returns':
        return _helper('done')
      case 'dev-additional-spending-tilt':
        return _helper('done')
      case 'dev-time':
        return _helper('done')
      default:
        noCase(type)
    }
  }
}
