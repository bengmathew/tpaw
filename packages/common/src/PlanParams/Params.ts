import { Params20 } from './Params20'

export type Params = Params20.Params
export type PlanParams = Params20.PlanParams
export type NonPlanParams = Params20.NonPlanParams
export type CalendarMonth = Params20.CalendarMonth
export type InMonths = Params20.InMonths
export type Person = Params20.Person
export type People = Params20.People
export type Month = Params20.Month
export type MonthRange = Params20.MonthRange
export type ValueForMonthRange = Params20.ValueForMonthRange
export type LabeledAmount = Params20.LabeledAmount
export type GlidePath = Params20.GlidePath

export const planParamsGuard = Params20.guard
export const MAX_AGE_IN_MONTHS = Params20.MAX_AGE_IN_MONTHS
export const MAX_LABEL_LENGTH = Params20.MAX_LABEL_LENGTH
export const MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY =
  Params20.MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY
export const MAX_SIZE_FOR_MONTH_RANGE_ARR =
  Params20.MAX_SIZE_FOR_MONTH_RANGE_ARR
export const MAX_EXTERNAL_LEGACY_SOURCES = Params20.MAX_EXTERNAL_LEGACY_SOURCES
export const RISK_TOLERANCE_VALUES = Params20.RISK_TOLERANCE_VALUES
export const TIME_PREFERENCE_VALUES = Params20.TIME_PREFERENCE_VALUES
export const ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES =
  Params20.ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES
export const SPAW_ANNUAL_SPENDING_TILT_VALUES =
  Params20.SPAW_ANNUAL_SPENDING_TILT_VALUES
export const MANUAL_INFLATION_VALUES = Params20.MANUAL_INFLATION_VALUES
export const MANUAL_STOCKS_BONDS_RETURNS_VALUES =
  Params20.MANUAL_STOCKS_BONDS_RETURNS_VALUES
export const MIN_PLAN_PARAM_TIME = Params20.MIN_PLAN_PARAM_TIME
export const calendarMonthFromTime = Params20.calendarMonthFromTime
