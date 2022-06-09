import {TPAWParamsV7} from './TPAWParamsV7'

export type TPAWParams = TPAWParamsV7.Params
export type Person = TPAWParamsV7.Person
export type People = TPAWParamsV7.People
export type Year = TPAWParamsV7.Year
export type YearRange = TPAWParamsV7.YearRange
export type ValueForYearRange = TPAWParamsV7.ValueForYearRange
export type LabeledAmount = TPAWParamsV7.LabeledAmount
export type GlidePath = TPAWParamsV7.GlidePath

export const tpawParamsValidator = TPAWParamsV7.validator
export const MAX_AGE = TPAWParamsV7.MAX_AGE
export const MAX_LABEL_LENGTH = TPAWParamsV7.MAX_LABEL_LENGTH
