import { PlanParams19 } from './PlanParams19'

export type PlanParams = PlanParams19.Params
export type Person = PlanParams19.Person
export type People = PlanParams19.People
export type Month = PlanParams19.Month
export type MonthRange = PlanParams19.MonthRange
export type ValueForMonthRange = PlanParams19.ValueForMonthRange
export type LabeledAmount = PlanParams19.LabeledAmount
export type GlidePath = PlanParams19.GlidePath

export const planParamsGuard = PlanParams19.guard
export const MAX_AGE_IN_MONTHS = PlanParams19.MAX_AGE_IN_MONTHS
export const MAX_LABEL_LENGTH = PlanParams19.MAX_LABEL_LENGTH
export const MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY =
  PlanParams19.MAX_SIZE_FOR_GLIDE_PATH_INTERMEDIATE_ARRAY
export const MAX_SIZE_FOR_MONTH_RANGE_ARR =
  PlanParams19.MAX_SIZE_FOR_MONTH_RANGE_ARR
export const MAX_EXTERNAL_LEGACY_SOURCES =
  PlanParams19.MAX_EXTERNAL_LEGACY_SOURCES
export const RISK_TOLERANCE_VALUES = PlanParams19.RISK_TOLERANCE_VALUES
export const TIME_PREFERENCE_VALUES = PlanParams19.TIME_PREFERENCE_VALUES
export const ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES =
  PlanParams19.ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES
export const SPAW_ANNUAL_SPENDING_TILT_VALUES =
  PlanParams19.SPAW_ANNUAL_SPENDING_TILT_VALUES
export const MANUAL_INFLATION_VALUES = PlanParams19.MANUAL_INFLATION_VALUES
export const MANUAL_STOCKS_BONDS_RETURNS_VALUES =
  PlanParams19.MANUAL_STOCKS_BONDS_RETURNS_VALUES
