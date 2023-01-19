import { PlanParams16 } from './PlanParams16'

export type PlanParams = PlanParams16.Params
export type Person = PlanParams16.Person
export type People = PlanParams16.People
export type Year = PlanParams16.Year
export type YearRange = PlanParams16.YearRange
export type ValueForYearRange = PlanParams16.ValueForYearRange
export type LabeledAmount = PlanParams16.LabeledAmount
export type GlidePath = PlanParams16.GlidePath

export const planParamsGuard = PlanParams16.guard
export const MAX_AGE = PlanParams16.MAX_AGE
export const MAX_LABEL_LENGTH = PlanParams16.MAX_LABEL_LENGTH
export const MAX_NUM_YEARS_IN_GLIDE_PATH =
  PlanParams16.MAX_NUM_YEARS_IN_GLIDE_PATH
export const MAX_VALUE_FOR_YEAR_RANGE = PlanParams16.MAX_VALUE_FOR_YEAR_RANGE
export const MAX_EXTERNAL_LEGACY_SOURCES =
  PlanParams16.MAX_EXTERNAL_LEGACY_SOURCES
export const RISK_TOLERANCE_VALUES = PlanParams16.RISK_TOLERANCE_VALUES
export const TIME_PREFERENCE_VALUES = PlanParams16.TIME_PREFERENCE_VALUES
export const SPAW_SPENDING_TILT_VALUES = PlanParams16.SPAW_SPENDING_TILT_VALUES
export const MANUAL_INFLATION_VALUES = PlanParams16.MANUAL_INFLATION_VALUES
export const MANUAL_STOCKS_BONDS_RETURNS_VALUES =
  PlanParams16.MANUAL_STOCKS_BONDS_RETURNS_VALUES

