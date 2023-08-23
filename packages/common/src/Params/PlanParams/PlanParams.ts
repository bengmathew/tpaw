import { PlanParams21 } from './PlanParams21'

type _GetVersion<T> = T extends { v: infer V } ? V : 1

export type PlanParams = PlanParams21.PlanParams
export type SomePlanParams = PlanParams21.SomePlanParams
export type SomePlanParamsVersion = _GetVersion<SomePlanParams>
export type CalendarMonth = PlanParams21.CalendarMonth
export type InMonths = PlanParams21.InMonths
export type Person = PlanParams21.Person
export type People = PlanParams21.People
export type Month = PlanParams21.Month
export type MonthRange = PlanParams21.MonthRange
export type ValueForMonthRange = PlanParams21.ValueForMonthRange
export type ValueForMonthRanges = PlanParams21.ValueForMonthRanges
export type LabeledAmount = PlanParams21.LabeledAmount
export type GlidePath = PlanParams21.GlidePath

export const currentPlanParamsVersion = PlanParams21.currentVersion
export const planParamsComponentGuards = PlanParams21.componentGuards
export const planParamsGuard = PlanParams21.guard
export const planParamsBackwardsCompatibleGuard =
  PlanParams21.backwardsCompatibleGuard
export const planParamsMigrate = PlanParams21.migrate
export const MAX_AGE_IN_MONTHS = PlanParams21.MAX_AGE_IN_MONTHS
export const MAX_LABEL_LENGTH = PlanParams21.MAX_LABEL_LENGTH
export const MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY =
  PlanParams21.MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY
export const MAX_SIZE_FOR_MONTH_RANGE_ARR =
  PlanParams21.MAX_SIZE_FOR_MONTH_RANGE_ARR
export const MAX_EXTERNAL_LEGACY_SOURCES =
  PlanParams21.MAX_EXTERNAL_LEGACY_SOURCES
export const RISK_TOLERANCE_VALUES = PlanParams21.RISK_TOLERANCE_VALUES
export const TIME_PREFERENCE_VALUES = PlanParams21.TIME_PREFERENCE_VALUES
export const ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES =
  PlanParams21.ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES
export const SPAW_ANNUAL_SPENDING_TILT_VALUES =
  PlanParams21.SPAW_ANNUAL_SPENDING_TILT_VALUES
export const MANUAL_INFLATION_VALUES = PlanParams21.MANUAL_INFLATION_VALUES
export const MANUAL_STOCKS_BONDS_RETURNS_VALUES =
  PlanParams21.MANUAL_STOCKS_BONDS_RETURNS_VALUES
export const MIN_PLAN_PARAM_TIME = PlanParams21.MIN_PLAN_PARAM_TIME
export const calendarMonthFromTime = PlanParams21.calendarMonthFromTime
