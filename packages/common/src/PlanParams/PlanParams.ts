import { PlanParams15 } from './PlanParams15'

export type PlanParams = PlanParams15.Params
export type Person = PlanParams15.Person
export type People = PlanParams15.People
export type Year = PlanParams15.Year
export type YearRange = PlanParams15.YearRange
export type ValueForYearRange = PlanParams15.ValueForYearRange
export type LabeledAmount = PlanParams15.LabeledAmount
export type GlidePath = PlanParams15.GlidePath

export const planParamsGuard = PlanParams15.guard
export const MAX_AGE = PlanParams15.MAX_AGE
export const MAX_LABEL_LENGTH = PlanParams15.MAX_LABEL_LENGTH
export const MAX_NUM_YEARS_IN_GLIDE_PATH =
  PlanParams15.MAX_NUM_YEARS_IN_GLIDE_PATH
export const MAX_VALUE_FOR_YEAR_RANGE = PlanParams15.MAX_VALUE_FOR_YEAR_RANGE
export const MAX_EXTERNAL_LEGACY_SOURCES =
  PlanParams15.MAX_EXTERNAL_LEGACY_SOURCES
export const RISK_TOLERANCE_VALUES = PlanParams15.RISK_TOLERANCE_VALUES
export const TIME_PREFERENCE_VALUES = PlanParams15.TIME_PREFERENCE_VALUES
export const SPAW_SPENDING_TILT_VALUES = PlanParams15.SPAW_SPENDING_TILT_VALUES
export const MANUAL_INFLATION_VALUES = PlanParams15.MANUAL_INFLATION_VALUES
export const MANUAL_STOCKS_BONDS_RETURNS_VALUES =
  PlanParams15.MANUAL_STOCKS_BONDS_RETURNS_VALUES

export const planParamsTest = PlanParams15.test
