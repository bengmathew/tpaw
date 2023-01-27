import { PlanParams18 } from './PlanParams18'

export type PlanParams = PlanParams18.Params
export type Person = PlanParams18.Person
export type People = PlanParams18.People
export type Year = PlanParams18.Year
export type YearRange = PlanParams18.YearRange
export type ValueForYearRange = PlanParams18.ValueForYearRange
export type LabeledAmount = PlanParams18.LabeledAmount
export type GlidePath = PlanParams18.GlidePath

export const planParamsGuard = PlanParams18.guard
export const MAX_AGE = PlanParams18.MAX_AGE
export const MAX_LABEL_LENGTH = PlanParams18.MAX_LABEL_LENGTH
export const MAX_NUM_YEARS_IN_GLIDE_PATH =
  PlanParams18.MAX_NUM_YEARS_IN_GLIDE_PATH
export const MAX_VALUE_FOR_YEAR_RANGE = PlanParams18.MAX_VALUE_FOR_YEAR_RANGE
export const MAX_EXTERNAL_LEGACY_SOURCES =
  PlanParams18.MAX_EXTERNAL_LEGACY_SOURCES
export const RISK_TOLERANCE_VALUES = PlanParams18.RISK_TOLERANCE_VALUES
export const TIME_PREFERENCE_VALUES = PlanParams18.TIME_PREFERENCE_VALUES
export const ADDITIONAL_SPENDING_TILT_VALUES =
  PlanParams18.ADDITIONAL_SPENDING_TILT_VALUES
export const SPAW_SPENDING_TILT_VALUES = PlanParams18.SPAW_SPENDING_TILT_VALUES
export const MANUAL_INFLATION_VALUES = PlanParams18.MANUAL_INFLATION_VALUES
export const MANUAL_STOCKS_BONDS_RETURNS_VALUES =
  PlanParams18.MANUAL_STOCKS_BONDS_RETURNS_VALUES
