import { assert, getDefaultPlanParams, TPAWRiskLevel } from '@tpaw/common'
import { noCase } from '../Utils/Utils'

export const riskLevelLabel = (riskLevel: TPAWRiskLevel) => {
  const label = (() => {
    switch (riskLevel) {
      case 'riskLevel-1':
        return 'Conservative'
      case 'riskLevel-2':
        return 'Moderately Conservative'
      case 'riskLevel-3':
        return 'Moderately Aggressive'
      case 'riskLevel-4':
        return 'Aggressive'
      default:
        noCase(riskLevel)
    }
  })()
  return riskLevel === _defaultTPAWRiskLevel ? `${label}` : label
}

const _defaultTPAWRiskLevel = (() => {
  const risk = getDefaultPlanParams().risk
  assert(risk.useTPAWPreset)
  return risk.tpawPreset
})()
