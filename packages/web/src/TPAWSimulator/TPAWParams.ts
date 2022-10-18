import {TPAWParamsV14} from './TPAWParamsV14'

export type TPAWParams = TPAWParamsV14.Params
export type Person = TPAWParamsV14.Person
export type People = TPAWParamsV14.People
export type Year = TPAWParamsV14.Year
export type YearRange = TPAWParamsV14.YearRange
export type ValueForYearRange = TPAWParamsV14.ValueForYearRange
export type LabeledAmount = TPAWParamsV14.LabeledAmount
export type GlidePath = TPAWParamsV14.GlidePath
export type TPAWRiskLevel = TPAWParamsV14.TPAWRiskLevel
export type TPAWRisk = TPAWParamsV14.TPAWRisk

export const tpawParamsValidator = TPAWParamsV14.validator
export const MAX_AGE = TPAWParamsV14.MAX_AGE
export const MAX_LABEL_LENGTH = TPAWParamsV14.MAX_LABEL_LENGTH
