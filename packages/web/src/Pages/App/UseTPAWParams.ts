import _ from 'lodash'
import {useRouter} from 'next/dist/client/router'
import {useEffect, useState} from 'react'
import {getDefaultParams} from '../../TPAWSimulator/DefaultParams'
import {
  TPAWParams,
  tpawParamsValidator,
  TPAWParamsWithoutHistorical,
  ValueForYearRange,
} from '../../TPAWSimulator/TPAWParams'
import {numericYear} from '../../TPAWSimulator/TPAWParamsProcessed'
import {TPAWParamsV1WithoutHistorical} from '../../TPAWSimulator/TPAWParamsV1'
import {tpawParamsV1Validator} from '../../TPAWSimulator/TPAWParamsV1Validator'
import {TPAWParamsV2WithoutHistorical} from '../../TPAWSimulator/TPAWParamsV2'
import {tpawParamsV2Validator} from '../../TPAWSimulator/TPAWParamsV2Validator'
import {TPAWParamsV3WithoutHistorical} from '../../TPAWSimulator/TPAWParamsV3'
import {tpawParamsV3Validator} from '../../TPAWSimulator/TPAWParamsV3Validator'
import {TPAWParamsV4WithoutHistorical} from '../../TPAWSimulator/TPAWParamsV4'
import {Validator} from '../../Utils/Validator'
import {AppError} from './AppError'

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
  const [history, setHistory] = useState(() => ({
    stack: [
      _parseExternalParams(router.query['params']) ??
        _parseExternalParams(window.localStorage.getItem('params')) ??
        getDefaultParams(),
    ],
    curr: 0,
  }))

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
        setHistory(e.shiftKey ? _redo : _undo)
      }
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [])

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

export const tpawParamsForURL = (params: TPAWParams) =>
  JSON.stringify(_removeHistorical(params))

function _parseExternalParams(str: string | string[] | undefined | null) {
  if (typeof str !== 'string') return null
  try {
    const parsed = JSON.parse(str)
    try {
      let v4: TPAWParamsWithoutHistorical
      if (parsed.v === 4) {
        v4 = tpawParamsValidator(parsed)
      } else if (parsed.v === 3) {
        v4 = _v3ToV4(tpawParamsV3Validator(parsed))
      } else if (parsed.v === 2) {
        v4 = _v3ToV4(_v2ToV3(tpawParamsV2Validator(parsed)))
      } else {
        v4 = _v3ToV4(_v2ToV3(_v1ToV2(tpawParamsV1Validator(parsed))))
      }
      return _addHistorical(v4)
    } catch (e) {
      if (e instanceof Validator.Failed) {
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

const _removeHistorical = (params: TPAWParams): TPAWParamsWithoutHistorical => {
  const p: any = _.cloneDeep(params)
  delete p.returns.historical
  return p
}

const _addHistorical = (params: TPAWParamsWithoutHistorical): TPAWParams => ({
  ...params,
  returns: {
    ...params.returns,
    historical: {
      adjust: {
        type: 'to',
        stocks: params.returns.expected.stocks,
        bonds: params.returns.expected.bonds,
      },
    },
  },
})

const _v1ToV2 = (
  v1: TPAWParamsV1WithoutHistorical
): TPAWParamsV2WithoutHistorical => {
  type ValueForYearRange = TPAWParamsV2WithoutHistorical['savings'][number]
  const savings: ValueForYearRange[] = []
  const retirementIncome: ValueForYearRange[] = []
  v1.savings.forEach(x => {
    const start = numericYear(v1, x.yearRange.start)
    const end = numericYear(v1, x.yearRange.end)
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
): TPAWParamsV4WithoutHistorical => {
  const {retirementIncome, savings, withdrawals, ...rest} = _.cloneDeep(v3)
  const addId = (
    x: TPAWParamsV3WithoutHistorical['savings'][number],
    id: number
  ): ValueForYearRange => ({...x, id})
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
