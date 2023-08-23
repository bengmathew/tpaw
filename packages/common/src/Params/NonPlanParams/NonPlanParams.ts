import { NonPlanParams1 } from './NonPlanParams21'

export type NonPlanParams = NonPlanParams1.NonPlanParams
export type SomeNonPlanParams = NonPlanParams1.SomeNonPlanParams

export const currentNonPlanParamsVersion = NonPlanParams1.currentVersion
export const nonPlanParamsGuard = NonPlanParams1.guard
export const nonPlanParamsMigrate = NonPlanParams1.migrate
export const nonPlanParamsBackwardsCompatibleGuard =
  NonPlanParams1.backwardsCompatibleGuard
