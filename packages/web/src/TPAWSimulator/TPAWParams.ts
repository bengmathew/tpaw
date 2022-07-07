import {TPAWParamsV9} from './TPAWParamsV9'

export type TPAWParams = TPAWParamsV9.Params
export type Person = TPAWParamsV9.Person
export type People = TPAWParamsV9.People
export type Year = TPAWParamsV9.Year
export type YearRange = TPAWParamsV9.YearRange
export type ValueForYearRange = TPAWParamsV9.ValueForYearRange
export type LabeledAmount = TPAWParamsV9.LabeledAmount
export type GlidePath = TPAWParamsV9.GlidePath

export const tpawParamsValidator = TPAWParamsV9.validator
export const MAX_AGE = TPAWParamsV9.MAX_AGE
export const MAX_LABEL_LENGTH = TPAWParamsV9.MAX_LABEL_LENGTH
