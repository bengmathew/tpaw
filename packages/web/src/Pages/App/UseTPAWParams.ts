import _ from 'lodash'
import {useRouter} from 'next/dist/client/router'
import {useEffect, useState} from 'react'
import {getDefaultParams} from '../../TPAWSimulator/DefaultParams'
import {historicalReturns} from '../../TPAWSimulator/HistoricalReturns'
import {TPAWParams} from '../../TPAWSimulator/TPAWParams'
import {tpawParamsValidator} from '../../TPAWSimulator/TPAWParamsValidator'
import {StateObj} from '../../Utils/UseStateObj'
import {Validator} from '../../Utils/Validator'
import { Config } from '../Config'
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

export function useTPAWParams(): StateObj<TPAWParams> {
  const router = useRouter()
  const [history, setHistory] = useState(() => ({
    stack: [
      _parseURLParams(router.query['params']) ??
        _parseURLParams(window.localStorage.getItem('params')) ??
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
    value,
    set: (params: TPAWParams | ((params: TPAWParams) => TPAWParams)) => {
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

function _parseURLParams(str: string | string[] | undefined | null) {
  if (typeof str !== 'string') return null
  try {
    const parsed = JSON.parse(str)
    try {
      return _addHistorical(tpawParamsValidator(parsed))
    } catch (e) {
      if (e instanceof Validator.Failed) {
        throw new AppError(`Error in parameter: ${e.fullMessage}`)
      } else {
        throw e
      }
    }
  } catch (e) {
    if (e instanceof SyntaxError) {
      throw new AppError('URL is not well formatted.')
    } else {
      throw e
    }
  }
}

type _WithoutHistorical = Omit<TPAWParams, 'returns'> & {
  returns: Omit<TPAWParams['returns'], 'historical'>
}

const _removeHistorical = (params: TPAWParams): _WithoutHistorical => {
  const p: any = _.cloneDeep(params)
  delete p.returns.historical
  return p
}

const _addHistorical = (params: _WithoutHistorical): TPAWParams => ({
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
