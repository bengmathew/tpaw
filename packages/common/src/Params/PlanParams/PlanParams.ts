import { PlanParams22 } from './PlanParams22'

type _GetVersion<T> = T extends { v: infer V } ? V : 1

export type PlanParams = PlanParams22.PlanParams
export type SomePlanParams = PlanParams22.SomePlanParams
export type SomePlanParamsVersion = _GetVersion<SomePlanParams>
export type CalendarMonth = PlanParams22.CalendarMonth
export type InMonths = PlanParams22.InMonths
export type Person = PlanParams22.Person
export type People = PlanParams22.People
export type Month = PlanParams22.Month
export type MonthRange = PlanParams22.MonthRange
export type ValueForMonthRange = PlanParams22.ValueForMonthRange
export type ValueForMonthRanges = PlanParams22.ValueForMonthRanges
export type LabeledAmount = PlanParams22.LabeledAmount
export type GlidePath = PlanParams22.GlidePath

export const currentPlanParamsVersion = PlanParams22.currentVersion
export const planParamsComponentGuards = PlanParams22.componentGuards
export const planParamsGuard = PlanParams22.guard
export const planParamsBackwardsCompatibleGuard =
  PlanParams22.backwardsCompatibleGuard
export const planParamsMigrate = PlanParams22.migrate
export const MAX_AGE_IN_MONTHS = PlanParams22.MAX_AGE_IN_MONTHS
export const MAX_LABEL_LENGTH = PlanParams22.MAX_LABEL_LENGTH
export const MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY =
  PlanParams22.MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY
export const MAX_SIZE_FOR_MONTH_RANGE_ARR =
  PlanParams22.MAX_SIZE_FOR_MONTH_RANGE_ARR
export const MAX_EXTERNAL_LEGACY_SOURCES =
  PlanParams22.MAX_EXTERNAL_LEGACY_SOURCES
export const RISK_TOLERANCE_VALUES = PlanParams22.RISK_TOLERANCE_VALUES
export const TIME_PREFERENCE_VALUES = PlanParams22.TIME_PREFERENCE_VALUES
export const ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES =
  PlanParams22.ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES
export const SPAW_ANNUAL_SPENDING_TILT_VALUES =
  PlanParams22.SPAW_ANNUAL_SPENDING_TILT_VALUES
export const MANUAL_INFLATION_VALUES = PlanParams22.MANUAL_INFLATION_VALUES
export const MANUAL_STOCKS_BONDS_RETURNS_VALUES =
  PlanParams22.MANUAL_STOCKS_BONDS_RETURNS_VALUES
export const MIN_PLAN_PARAM_TIME = PlanParams22.MIN_PLAN_PARAM_TIME
export const calendarMonthFromTime = PlanParams22.calendarMonthFromTime
