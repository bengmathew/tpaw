import {TPAWParamsV10} from './TPAWParamsV10'

export type TPAWParams = TPAWParamsV10.Params
export type Person = TPAWParamsV10.Person
export type People = TPAWParamsV10.People
export type Year = TPAWParamsV10.Year
export type YearRange = TPAWParamsV10.YearRange
export type ValueForYearRange = TPAWParamsV10.ValueForYearRange
export type LabeledAmount = TPAWParamsV10.LabeledAmount
export type GlidePath = TPAWParamsV10.GlidePath

export const tpawParamsValidator = TPAWParamsV10.validator
export const MAX_AGE = TPAWParamsV10.MAX_AGE
export const MAX_LABEL_LENGTH = TPAWParamsV10.MAX_LABEL_LENGTH
