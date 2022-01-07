import _ from 'lodash'
import {useRouter} from 'next/dist/client/router'
import {useEffect, useState} from 'react'
import {getDefaultParams} from '../../TPAWSimulator/DefaultParams'
import {
  numericYear,
  TPAWParams,
  TPAWParamsWithoutHistorical,
  ValueForYearRange,
} from '../../TPAWSimulator/TPAWParams'
import {TPAWParamsV1WithoutHistorical} from '../../TPAWSimulator/TPAWParamsV1'
import {tpawParamsV1Validator} from '../../TPAWSimulator/TPAWParamsV1Validator'
import {tpawParamsValidator} from '../../TPAWSimulator/TPAWParamsValidator'
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
      let v2: TPAWParamsWithoutHistorical
      if (parsed.v === 2) {
        v2 = tpawParamsValidator(parsed)
      } else {
        v2 = _v1ToV2(tpawParamsV1Validator(parsed))
      }
      return _addHistorical(v2)
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
): TPAWParamsWithoutHistorical => {
  const savings: ValueForYearRange[] = []
  const retirementIncome: ValueForYearRange[] = []
  v1.savings.forEach(x => {
    const start = numericYear(v1, x.yearRange.start)
    const end = numericYear(v1, x.yearRange.end)
    if (start < v1.age.retirement && end >= v1.age.retirement) {
      savings.push({
        ...x,
        yearRange: {...x.yearRange, end: 'retirement' as const},
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
