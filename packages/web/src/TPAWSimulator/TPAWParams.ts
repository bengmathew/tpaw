import {TPAWParamsV5} from './TPAWParamsV5'

export type TPAWParams = TPAWParamsV5.Params
export type TPAWParamsWithoutHistorical = TPAWParamsV5.ParamsWithoutHistorical
export type Person = TPAWParamsV5.Person
export type People = TPAWParamsV5.People
export type Year = TPAWParamsV5.Year
export type YearRange = TPAWParamsV5.YearRange
export type ValueForYearRange = TPAWParamsV5.ValueForYearRange
export type LabeledAmount = TPAWParamsV5.LabeledAmount

export const tpawParamsValidator = TPAWParamsV5.validator
export const MAX_AGE = TPAWParamsV5.MAX_AGE
export const MAX_LABEL_LENGTH = TPAWParamsV5.MAX_LABEL_LENGTH
