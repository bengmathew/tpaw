import {TPAWParamsV4, TPAWParamsV4WithoutHistorical} from './TPAWParamsV4'
import {MAX_AGE_V4, tpawParamsV4Validator} from './TPAWParamsV4Validator'

export type TPAWParams = TPAWParamsV4
export type TPAWParamsWithoutHistorical = TPAWParamsV4WithoutHistorical
export type ValueForYearRange = TPAWParams['savings'][number]
export type YearRange = ValueForYearRange['yearRange']
export type YearRangeEdge = YearRange['start']
export type YearRangeFixedEdge = Exclude<YearRangeEdge, number>

export const tpawParamsValidator = tpawParamsV4Validator
export const MAX_AGE = MAX_AGE_V4
