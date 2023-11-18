import { NonPlanParams22 } from './NonPlanParams22'

export type NonPlanParams = NonPlanParams22.NonPlanParams
export type SomeNonPlanParams = NonPlanParams22.SomeNonPlanParams

export const currentNonPlanParamsVersion = NonPlanParams22.currentVersion
export const nonPlanParamsGuard = NonPlanParams22.guard
export const nonPlanParamsMigrate = NonPlanParams22.migrate
export const nonPlanParamsBackwardsCompatibleGuard =
  NonPlanParams22.backwardsCompatibleGuard
