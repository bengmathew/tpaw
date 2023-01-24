import { PlanParams17 } from './PlanParams17'

export type PlanParams = PlanParams17.Params
export type Person = PlanParams17.Person
export type People = PlanParams17.People
export type Year = PlanParams17.Year
export type YearRange = PlanParams17.YearRange
export type ValueForYearRange = PlanParams17.ValueForYearRange
export type LabeledAmount = PlanParams17.LabeledAmount
export type GlidePath = PlanParams17.GlidePath

export const planParamsGuard = PlanParams17.guard
export const MAX_AGE = PlanParams17.MAX_AGE
export const MAX_LABEL_LENGTH = PlanParams17.MAX_LABEL_LENGTH
export const MAX_NUM_YEARS_IN_GLIDE_PATH =
  PlanParams17.MAX_NUM_YEARS_IN_GLIDE_PATH
export const MAX_VALUE_FOR_YEAR_RANGE = PlanParams17.MAX_VALUE_FOR_YEAR_RANGE
export const MAX_EXTERNAL_LEGACY_SOURCES =
  PlanParams17.MAX_EXTERNAL_LEGACY_SOURCES
export const RISK_TOLERANCE_VALUES = PlanParams17.RISK_TOLERANCE_VALUES
export const TIME_PREFERENCE_VALUES = PlanParams17.TIME_PREFERENCE_VALUES
export const SPAW_SPENDING_TILT_VALUES = PlanParams17.SPAW_SPENDING_TILT_VALUES
export const MANUAL_INFLATION_VALUES = PlanParams17.MANUAL_INFLATION_VALUES
export const MANUAL_STOCKS_BONDS_RETURNS_VALUES =
  PlanParams17.MANUAL_STOCKS_BONDS_RETURNS_VALUES
