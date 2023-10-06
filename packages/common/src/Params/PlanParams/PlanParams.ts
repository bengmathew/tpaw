import { PlanParams23 as PlanParamsCurr } from './PlanParams23'

type _GetVersion<T> = T extends { v: infer V } ? V : 1

export type PlanParams = PlanParamsCurr.PlanParams
export type SomePlanParams = PlanParamsCurr.SomePlanParams
export type SomePlanParamsVersion = _GetVersion<SomePlanParams>
export type CalendarMonth = PlanParamsCurr.CalendarMonth
export type InMonths = PlanParamsCurr.InMonths
export type Person = PlanParamsCurr.Person
export type People = PlanParamsCurr.People
export type Month = PlanParamsCurr.Month
export type MonthRange = PlanParamsCurr.MonthRange
export type ValueForMonthRange = PlanParamsCurr.ValueForMonthRange
export type ValueForMonthRanges = PlanParamsCurr.ValueForMonthRanges
export type LabeledAmount = PlanParamsCurr.LabeledAmount
export type GlidePath = PlanParamsCurr.GlidePath

export const currentPlanParamsVersion = PlanParamsCurr.currentVersion
export const planParamsComponentGuards = PlanParamsCurr.componentGuards
export const planParamsGuard = PlanParamsCurr.guard
export const planParamsBackwardsCompatibleGuard =
  PlanParamsCurr.backwardsCompatibleGuard
export const planParamsMigrate = PlanParamsCurr.migrate
export const MAX_AGE_IN_MONTHS = PlanParamsCurr.MAX_AGE_IN_MONTHS
export const MAX_LABEL_LENGTH = PlanParamsCurr.MAX_LABEL_LENGTH
export const MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY =
  PlanParamsCurr.MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY
export const MAX_SIZE_FOR_MONTH_RANGE_ARR =
  PlanParamsCurr.MAX_SIZE_FOR_MONTH_RANGE_ARR
export const MAX_EXTERNAL_LEGACY_SOURCES =
  PlanParamsCurr.MAX_EXTERNAL_LEGACY_SOURCES
export const STOCK_VOLATILITY_SCALE_VALUES =
  PlanParamsCurr.STOCK_VOLATILITY_SCALE_VALUES
export const RISK_TOLERANCE_VALUES = PlanParamsCurr.RISK_TOLERANCE_VALUES
export const TIME_PREFERENCE_VALUES = PlanParamsCurr.TIME_PREFERENCE_VALUES
export const ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES =
  PlanParamsCurr.ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES
export const SPAW_ANNUAL_SPENDING_TILT_VALUES =
  PlanParamsCurr.SPAW_ANNUAL_SPENDING_TILT_VALUES
export const MANUAL_INFLATION_VALUES = PlanParamsCurr.MANUAL_INFLATION_VALUES
export const MANUAL_STOCKS_BONDS_RETURNS_VALUES =
  PlanParamsCurr.MANUAL_STOCKS_BONDS_RETURNS_VALUES
export const MIN_PLAN_PARAM_TIME = PlanParamsCurr.MIN_PLAN_PARAM_TIME
export const calendarMonthFromTime = PlanParamsCurr.calendarMonthFromTime
