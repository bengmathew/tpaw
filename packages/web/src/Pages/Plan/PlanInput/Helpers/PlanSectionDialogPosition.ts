import { PlanParams, noCase } from '@tpaw/common'

export const planSectionDialogOrder: PlanParams['dialogPosition'][] = [
  'age',
  'current-portfolio-balance',
  'future-savings',
  'income-during-retirement',
  'show-results',
  'show-all-inputs',
  'done',
]
export function nextPlanSectionDialogPosition(
  section: Exclude<PlanParams['dialogPosition'], 'done'>,
  withdrawalStartMonthAMFN: number,
): PlanParams['dialogPosition']
export function nextPlanSectionDialogPosition(
  section: 'show-results' | 'show-all-inputs',
  withdrawalStartMonthAMFN?: number,
): PlanParams['dialogPosition']
export function nextPlanSectionDialogPosition(
  section: Exclude<PlanParams['dialogPosition'], 'done'>,
  withdrawalStartMonthAMFN?: number,
): PlanParams['dialogPosition'] {
  withdrawalStartMonthAMFN = withdrawalStartMonthAMFN ?? 0
  const order =
    withdrawalStartMonthAMFN <= 0
      ? planSectionDialogOrder.filter((x) => x !== 'future-savings')
      : planSectionDialogOrder
  return order[order.indexOf(section) + 1]
}

export const isPlanSectionDialogInOverlayMode = (
  dialogPosition: PlanParams['dialogPosition'],
): dialogPosition is 'show-results' | 'show-all-inputs' => {
  switch (dialogPosition) {
    case 'age':
    case 'current-portfolio-balance':
    case 'future-savings':
    case 'income-during-retirement':
    case 'done':
      return false
    case 'show-results':
    case 'show-all-inputs':
      return true
    default:
      noCase(dialogPosition)
  }
}
