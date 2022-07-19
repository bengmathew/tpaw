import _ from 'lodash'
import {useRouter} from 'next/dist/client/router'
import {useCallback, useEffect, useState} from 'react'
import {getDefaultParams} from '../../TPAWSimulator/DefaultParams'
import {TPAWParams} from '../../TPAWSimulator/TPAWParams'
import {TPAWParamsV1WithoutHistorical} from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV1'
import {tpawParamsV1Validator} from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV1Validator'
import {TPAWParamsV2WithoutHistorical} from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV2'
import {tpawParamsV2Validator} from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV2Validator'
import {TPAWParamsV3WithoutHistorical} from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV3'
import {tpawParamsV3Validator} from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV3Validator'
import {TPAWParamsV4} from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV4'
import {TPAWParamsV5} from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV5'
import {TPAWParamsV6} from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV6'
import {TPAWParamsV7} from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV7'
import {TPAWParamsV8} from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV8'

import {TPAWParamsV9} from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV9'
import { TPAWParamsV10 } from '../../TPAWSimulator/TPAWParamsOld/TPAWParamsV10'
import {useAssertConst} from '../../Utils/UseAssertConst'
import {fGet} from '../../Utils/Utils'
import {Validator} from '../../Utils/Validator'
import {AppError} from './AppError'
import { TPAWParamsV11 } from '../../TPAWSimulator/TPAWParamsV11'

type _History = {stack: TPAWParams[]; curr: number}
const _undo = (history: _History) =>
  history.curr === history.stack.length - 1
    ? history
    : {stack: history.stack, curr: history.curr + 1}

const _redo = (history: _History) =>
  history.curr === 0 ? history : {stack: history.stack, curr: history.curr - 1}

const _curr = ({stack, curr}: _History) => stack[curr]
const _new = ({stack, curr}: _History, params: TPAWParams) => ({
  stack: [params, ...stack.slice(curr)].slice(0, 100),
  curr: 0,
})

export function useTPAWParams() {
  const router = useRouter()
  const [abHistory, setABHistory] = useState(() => {
    const a = {
      stack: [
        _parseExternalParams(router.query['params']) ??
          _parseExternalParams(window.localStorage.getItem('params')) ??
          getDefaultParams(),
      ],
      curr: 0,
    }
    return {space: 'a' as 'a' | 'b', a, b: _.cloneDeep(a)}
  })
  const setParamSpace = useCallback(
    (space: 'a' | 'b') => setABHistory(x => ({...x, space})),
    []
  )

  const history = abHistory[abHistory.space]
  const setHistory = useCallback(
    (history: _History | ((x: _History) => _History)) => {
      setABHistory(abHistory => {
        const clone = _.cloneDeep(abHistory)
        clone[clone.space] =
          typeof history === 'function' ? history(clone[clone.space]) : history
        return clone
      })
    },
    []
  )

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
        setHistory(e.shiftKey ? _redo : _undo)
      }
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [setHistory])
  useAssertConst([setHistory])

  const value = _curr(history)
  useEffect(() => {
    if (typeof router.query['params'] === 'string') {
      const url = new URL(window.location.href)
      url.searchParams.set('params', tpawParamsForURL(value))
      void router.replace(url)
    }
    window.localStorage.setItem('params', tpawParamsForURL(value))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value])
  return {
    paramSpace: abHistory.space,
    setParamSpace,
    params: value,
    setParams: (params: TPAWParams | ((params: TPAWParams) => TPAWParams)) => {
      setHistory(history =>
        _new(
          history,
          typeof params === 'function' ? params(_curr(history)) : params
        )
      )
    },
  }
}

export const tpawParamsForURL = (params: TPAWParams) => JSON.stringify(params)

function _parseExternalParams(
  str: string | string[] | undefined | null
): TPAWParams | null {
  if (typeof str !== 'string') return null
  try {
    const parsed = JSON.parse(str)
    try {
      const v1 = !parsed.v ? tpawParamsV1Validator(parsed) : null
      const v2 =
        parsed.v === 2 ? tpawParamsV2Validator(parsed) : v1 ? _v1ToV2(v1) : null

      const v3 =
        parsed.v === 3 ? tpawParamsV3Validator(parsed) : v2 ? _v2ToV3(v2) : null
      const v4 =
        parsed.v === 4
          ? TPAWParamsV4.validator(parsed)
          : v3
          ? _v3ToV4(v3)
          : null
      const v5 =
        parsed.v === 5
          ? TPAWParamsV5.validator(parsed)
          : v4
          ? _v4ToV5(v4)
          : null

      const v6 =
        parsed.v === 6
          ? TPAWParamsV6.validator(parsed)
          : v5
          ? TPAWParamsV6.fromV5(v5)
          : null

      const v7 =
        parsed.v === 7
          ? TPAWParamsV7.validator(parsed)
          : v6
          ? TPAWParamsV7.fromV6(v6)
          : null

      const v8 =
        parsed.v === 8
          ? TPAWParamsV8.validator(parsed)
          : v7
          ? TPAWParamsV8.fromV7(v7)
          : null

      const v9 =
        parsed.v === 9
          ? TPAWParamsV9.validator(parsed)
          : v8
          ? TPAWParamsV9.fromV8(v8)
          : null

      const v10 =
        parsed.v === 10
          ? TPAWParamsV10.validator(parsed)
          : v9
          ? TPAWParamsV10.fromV9(v9)
          : null

      const v11 =
        parsed.v === 11
          ? TPAWParamsV11.validator(parsed)
          : TPAWParamsV11.fromV10(fGet(v10))

      return v11
    } catch (e) {
      if (e instanceof Validator.Failed) {
        console.dir(parsed)
        throw new AppError(`Error in parameter: ${e.fullMessage}`)
      } else {
        throw e
      }
    }
  } catch (e) {
    if (e instanceof SyntaxError) {
      throw new AppError('Parameters are not well formatted.')
    } else {
      throw e
    }
  }
}

const _v1ToV2 = (
  v1: TPAWParamsV1WithoutHistorical
): TPAWParamsV2WithoutHistorical => {
  type ValueForYearRange = TPAWParamsV2WithoutHistorical['savings'][number]
  const savings: ValueForYearRange[] = []
  const retirementIncome: ValueForYearRange[] = []
  v1.savings.forEach(x => {
    const start = _numericYear(v1, x.yearRange.start)
    const end = _numericYear(v1, x.yearRange.end)
    if (start < v1.age.retirement && end >= v1.age.retirement) {
      savings.push({
        ...x,
        yearRange: {...x.yearRange, end: 'lastWorkingYear' as const},
      })
      retirementIncome.push({
        ...x,
        yearRange: {...x.yearRange, start: 'retirement' as const},
      })
    } else {
      start < v1.age.retirement ? savings.push(x) : retirementIncome.push(x)
    }
  })

  return {
    v: 2,
    ...v1,
    savings,
    retirementIncome,
  }
}

const _v2ToV3 = (
  v2: TPAWParamsV2WithoutHistorical
): TPAWParamsV3WithoutHistorical => {
  return {
    ..._.cloneDeep(v2),
    v: 3,
    spendingFloor: null,
  }
}
const _v3ToV4 = (
  v3: TPAWParamsV3WithoutHistorical
): TPAWParamsV4.ParamsWithoutHistorical => {
  const {retirementIncome, savings, withdrawals, ...rest} = _.cloneDeep(v3)
  const addId = (
    x: TPAWParamsV3WithoutHistorical['savings'][number],
    id: number
  ): TPAWParamsV4.ValueForYearRange => ({...x, id})
  retirementIncome
  return {
    ...rest,
    v: 4,
    retirementIncome: retirementIncome.map(addId),
    savings: retirementIncome.map(addId),
    withdrawals: {
      fundedByBonds: withdrawals.fundedByBonds.map(addId),
      fundedByRiskPortfolio: withdrawals.fundedByRiskPortfolio.map(addId),
    },
  }
}
const _v4ToV5 = (
  v4: TPAWParamsV4.ParamsWithoutHistorical
): TPAWParamsV5.ParamsWithoutHistorical => {
  const {age, savings, retirementIncome, withdrawals, ...rest} = v4

  const year = (year: TPAWParamsV4.YearRangeEdge): TPAWParamsV5.Year =>
    year === 'start'
      ? {type: 'now'}
      : typeof year === 'number'
      ? {type: 'numericAge', person: 'person1', age: year}
      : {
          type: 'namedAge',
          person: 'person1',
          age: year === 'end' ? 'max' : year,
        }

  const valueForYearRange = ({
    yearRange,
    ...rest
  }: TPAWParamsV4.ValueForYearRange): TPAWParamsV5.ValueForYearRange => ({
    yearRange: {
      type: 'startAndEnd',
      start: year(yearRange.start),
      end: year(yearRange.end),
    },
    ...rest,
  })

  const result: TPAWParamsV5.ParamsWithoutHistorical = {
    ...rest,
    v: 5,
    people: {
      withPartner: false,
      person1: {
        ages:
          age.start === age.retirement
            ? {
                type: 'retired',
                current: age.start,
                max: age.end,
              }
            : {
                type: 'notRetired',
                current: age.start,
                retirement: age.retirement,
                max: age.end,
              },
        displayName: null,
      },
    },
    savings: savings.map(valueForYearRange),
    retirementIncome: retirementIncome.map(valueForYearRange),
    withdrawals: {
      fundedByBonds: withdrawals.fundedByBonds.flatMap(valueForYearRange),
      fundedByRiskPortfolio:
        withdrawals.fundedByRiskPortfolio.flatMap(valueForYearRange),
    },
  }

  TPAWParamsV5.validator(result)
  return result
}

const _numericYear = (
  {age}: {age: {start: number; retirement: number; end: number}},
  x: TPAWParamsV4.YearRangeEdge
) =>
  x === 'start'
    ? age.start
    : x === 'lastWorkingYear'
    ? age.retirement - 1
    : x === 'retirement'
    ? age.retirement
    : x === 'end'
    ? age.end
    : x
