import {TPAWParamsV8} from './TPAWParamsV8'

export type TPAWParams = TPAWParamsV8.Params
export type Person = TPAWParamsV8.Person
export type People = TPAWParamsV8.People
export type Year = TPAWParamsV8.Year
export type YearRange = TPAWParamsV8.YearRange
export type ValueForYearRange = TPAWParamsV8.ValueForYearRange
export type LabeledAmount = TPAWParamsV8.LabeledAmount
export type GlidePath = TPAWParamsV8.GlidePath

export const tpawParamsValidator = TPAWParamsV8.validator
export const MAX_AGE = TPAWParamsV8.MAX_AGE
export const MAX_LABEL_LENGTH = TPAWParamsV8.MAX_LABEL_LENGTH
