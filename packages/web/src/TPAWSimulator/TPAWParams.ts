import {TPAWParamsV3, TPAWParamsV3WithoutHistorical} from './TPAWParamsV3'
import {MAX_AGE_V3, tpawParamsV3Validator} from './TPAWParamsV3Validator'

export type TPAWParams = TPAWParamsV3
export type TPAWParamsWithoutHistorical = TPAWParamsV3WithoutHistorical
export type ValueForYearRange = TPAWParams['savings'][number]
export type YearRange = ValueForYearRange['yearRange']
export type YearRangeEdge = YearRange['start']
export type YearRangeFixedEdge = Exclude<YearRangeEdge, number>

export const tpawParamsValidator = tpawParamsV3Validator
export const MAX_AGE = MAX_AGE_V3
