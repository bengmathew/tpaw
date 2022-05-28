import {TPAWParamsV6} from './TPAWParamsV6'

export type TPAWParams = TPAWParamsV6.Params
export type TPAWParamsWithoutHistorical = TPAWParamsV6.ParamsWithoutHistorical
export type Person = TPAWParamsV6.Person
export type People = TPAWParamsV6.People
export type Year = TPAWParamsV6.Year
export type YearRange = TPAWParamsV6.YearRange
export type ValueForYearRange = TPAWParamsV6.ValueForYearRange
export type LabeledAmount = TPAWParamsV6.LabeledAmount
export type GlidePath = TPAWParamsV6.GlidePath

export const tpawParamsValidator = TPAWParamsV6.validator
export const MAX_AGE = TPAWParamsV6.MAX_AGE
export const MAX_LABEL_LENGTH = TPAWParamsV6.MAX_LABEL_LENGTH
