import { NonPlanParams26 as NonPlanParamsCurr } from './NonPlanParams26'

export type NonPlanParams = NonPlanParamsCurr.NonPlanParams
export type SomeNonPlanParams = NonPlanParamsCurr.SomeNonPlanParams

export const currentNonPlanParamsVersion = NonPlanParamsCurr.currentVersion
export const nonPlanParamsGuard = NonPlanParamsCurr.guard
export const nonPlanParamsMigrate = NonPlanParamsCurr.migrate
export const nonPlanParamsBackwardsCompatibleGuard =
  NonPlanParamsCurr.backwardsCompatibleGuard

export const nonPlanParamFns = NonPlanParamsCurr.fns
