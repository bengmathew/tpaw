import {TPAWParamsV13} from './TPAWParamsV13'

export type TPAWParams = TPAWParamsV13.Params
export type Person = TPAWParamsV13.Person
export type People = TPAWParamsV13.People
export type Year = TPAWParamsV13.Year
export type YearRange = TPAWParamsV13.YearRange
export type ValueForYearRange = TPAWParamsV13.ValueForYearRange
export type LabeledAmount = TPAWParamsV13.LabeledAmount
export type GlidePath = TPAWParamsV13.GlidePath
export type TPAWRiskLevel = TPAWParamsV13.TPAWRiskLevel
export type TPAWRisk = TPAWParamsV13.TPAWRisk

export const tpawParamsValidator = TPAWParamsV13.validator
export const MAX_AGE = TPAWParamsV13.MAX_AGE
export const MAX_LABEL_LENGTH = TPAWParamsV13.MAX_LABEL_LENGTH
