import {TPAWParamsV11} from './TPAWParamsV11'

export type TPAWParams = TPAWParamsV11.Params
export type Person = TPAWParamsV11.Person
export type People = TPAWParamsV11.People
export type Year = TPAWParamsV11.Year
export type YearRange = TPAWParamsV11.YearRange
export type ValueForYearRange = TPAWParamsV11.ValueForYearRange
export type LabeledAmount = TPAWParamsV11.LabeledAmount
export type GlidePath = TPAWParamsV11.GlidePath

export const tpawParamsValidator = TPAWParamsV11.validator
export const MAX_AGE = TPAWParamsV11.MAX_AGE
export const MAX_LABEL_LENGTH = TPAWParamsV11.MAX_LABEL_LENGTH
