import {TPAWParamsV12} from './TPAWParamsV12'

export type TPAWParams = TPAWParamsV12.Params
export type Person = TPAWParamsV12.Person
export type People = TPAWParamsV12.People
export type Year = TPAWParamsV12.Year
export type YearRange = TPAWParamsV12.YearRange
export type ValueForYearRange = TPAWParamsV12.ValueForYearRange
export type LabeledAmount = TPAWParamsV12.LabeledAmount
export type GlidePath = TPAWParamsV12.GlidePath

export const tpawParamsValidator = TPAWParamsV12.validator
export const MAX_AGE = TPAWParamsV12.MAX_AGE
export const MAX_LABEL_LENGTH = TPAWParamsV12.MAX_LABEL_LENGTH
