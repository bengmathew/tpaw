import { DialogPosition, PlanParams, noCase } from '@tpaw/common'

// type DialogPosition = PlanParams['dialogPositionNominal']
// export const planSectionDialogOrder: _DialogPosition[] = [
//   'age',
//   'current-portfolio-balance',
//   'future-savings',
//   'income-during-retirement',
//   'show-results',
//   'show-all-inputs',
//   'done',
// ]

// export function nextPlanSectionDialogPosition(
//   section: Exclude<_DialogPosition, 'done'>,
//   isFutureSavingsAllowed: boolean,
// ): _DialogPosition
// export function nextPlanSectionDialogPosition(
//   section: 'show-results' | 'show-all-inputs',
// ): _DialogPosition
// export function nextPlanSectionDialogPosition(
//   section: Exclude<_DialogPosition, 'done'>,
//   isFutureSavingsAllowed?: boolean,
// ): _DialogPosition {
//   const order =
//     isFutureSavingsAllowed ?? true
//       ? planSectionDialogOrder
//       : planSectionDialogOrder.filter((x) => x !== 'future-savings')
//   return order[order.indexOf(section) + 1]
// }

export const isPlanSectionDialogInOverlayMode = (
  dialogPosition: DialogPosition,
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
