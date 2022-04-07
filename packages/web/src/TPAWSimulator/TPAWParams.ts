import {V5Params} from './TPAWParamsV5'

export type TPAWParams = V5Params.Params
export type TPAWParamsWithoutHistorical = V5Params.ParamsWithoutHistorical
export type Person = V5Params.Person
export type People = V5Params.People
export type Year = V5Params.Year
export type YearRange = V5Params.YearRange
export type ValueForYearRange = V5Params.ValueForYearRange
export type LabeledAmount = V5Params.LabeledAmount

export const tpawParamsValidator = V5Params.validator
export const MAX_AGE = V5Params.MAX_AGE
export const MAX_LABEL_LENGTH = V5Params.MAX_LABEL_LENGTH
