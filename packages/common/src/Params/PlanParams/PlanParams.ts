import { PlanParams29 as PlanParamsCurr } from './PlanParams29'

type _GetVersion<T> = T extends { v: infer V } ? V : 1

export type PlanParams = PlanParamsCurr.PlanParams
export type SomePlanParams = PlanParamsCurr.SomePlanParams
export type SomePlanParamsVersion = _GetVersion<SomePlanParams>
export type TimestampedPlanParams = PlanParamsCurr.TimestampedPlanParams
export type DialogPosition = PlanParamsCurr.DialogPosition
export type CalendarMonth = PlanParamsCurr.CalendarMonth
export type CalendarDay = PlanParamsCurr.CalendarDay
export type InMonths = PlanParamsCurr.InMonths
export type PersonId = PlanParamsCurr.PersonId
export type Person = PlanParamsCurr.Person
export type People = PlanParamsCurr.People
export type Month = PlanParamsCurr.Month
export type MonthRange = PlanParamsCurr.MonthRange
export type LabeledAmountTimed = PlanParamsCurr.LabeledAmountTimed
export type LabeledAmountTimedList = PlanParamsCurr.LabeledAmountTimedList
export type LabeledAmountUntimed = PlanParamsCurr.LabeledAmountUntimed
export type LabeledAmountUntimedList = PlanParamsCurr.LabeledAmountUntimedList
export type LabeledAmountTimedOrUntimed =
  PlanParamsCurr.LabeledAmountTimedOrUntimed
export type GlidePath = PlanParamsCurr.GlidePath
export type LabeledAmountTimedLocation =
  PlanParamsCurr.LabeledAmountTimedLocation
export type LabeledAmountUntimedLocation =
  PlanParamsCurr.LabeledAmountUntimedLocation
export type LabeledAmountTimedOrUntimedLocation =
  PlanParamsCurr.LabeledAmountTimedOrUntimedLocation
export type GlidePathLocation = PlanParamsCurr.GlidePathLocation
export type MonthLocation = PlanParamsCurr.MonthLocation

export const currentPlanParamsVersion = PlanParamsCurr.currentVersion
export const planParamsComponentGuards = PlanParamsCurr.componentGuards
export const planParamsGuard = PlanParamsCurr.guard
export const planParamsBackwardsCompatibleGuard =
  PlanParamsCurr.backwardsCompatibleGuard
export const planParamsBackwardsUpToTimestampCompatibleGuard =
  PlanParamsCurr.backwardsCompatibleToTimestampGuard
export const planParamsMigrate = PlanParamsCurr.migrate
export const PLAN_PARAMS_CONSTANTS = PlanParamsCurr.CONSTANTS
