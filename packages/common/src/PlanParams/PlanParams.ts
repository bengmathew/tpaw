import { PlanParams14 } from './PlanParams14'

export type PlanParams = PlanParams14.Params
export type Person = PlanParams14.Person
export type People = PlanParams14.People
export type Year = PlanParams14.Year
export type YearRange = PlanParams14.YearRange
export type ValueForYearRange = PlanParams14.ValueForYearRange
export type LabeledAmount = PlanParams14.LabeledAmount
export type GlidePath = PlanParams14.GlidePath
export type TPAWRiskLevel = PlanParams14.TPAWRiskLevel
export type TPAWRisk = PlanParams14.TPAWRisk

export const planParamsGuard = PlanParams14.guard
export const MAX_AGE = PlanParams14.MAX_AGE
export const MAX_LABEL_LENGTH = PlanParams14.MAX_LABEL_LENGTH
export const MAX_NUM_YEARS_IN_GLIDE_PATH =
  PlanParams14.MAX_NUM_YEARS_IN_GLIDE_PATH
export const MAX_VALUE_FOR_YEAR_RANGE = PlanParams14.MAX_VALUE_FOR_YEAR_RANGE
export const MAX_EXTERNAL_LEGACY_SOURCES =
  PlanParams14.MAX_EXTERNAL_LEGACY_SOURCES
