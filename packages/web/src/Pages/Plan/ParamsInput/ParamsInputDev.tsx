import {faMinus, faPlus} from '@fortawesome/pro-light-svg-icons'
import {faCircle as faCircleRegular} from '@fortawesome/pro-regular-svg-icons'
import {faCircle as faCircleSelected} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {useEffect, useState} from 'react'
import {paddingCSS} from '../../../Utils/Geometry'
import {assert} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {useAmountInputState} from '../../Common/Inputs/AmountInput'
import {ToggleSwitch} from '../../Common/Inputs/ToggleSwitch'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputDev = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()

    const valueState = useAmountInputState(params.savingsAtStartOfStartYear)
    return (
      <ParamsInputBody {...props} headingMarginLeft="normal">
        <div className="">
          <div
            className="params-card"
            style={{padding: paddingCSS(props.sizing.cardPadding)}}
          >
            <h2 className="font-bold text-lg">Historical Returns</h2>

            <div className="mt-4">
              <h2
                className={`cursor-pointer `}
                onClick={() =>
                  setParams(p => {
                    const clone = _.cloneDeep(p)
                    clone.returns.historical = {
                      type: 'default',
                      adjust: {type: 'toExpected'},
                    }
                    return clone
                  })
                }
              >
                <FontAwesomeIcon
                  className="mr-2"
                  icon={
                    params.returns.historical.type === 'default'
                      ? faCircleSelected
                      : faCircleRegular
                  }
                />{' '}
                Real Historical Returns
              </h2>
            </div>
            <div className="mt-4">
              <h2
                className={`cursor-pointer `}
                onClick={() =>
                  setParams(p => {
                    const clone = _.cloneDeep(p)
                    clone.returns.historical = {
                      type: 'fixed',
                      stocks: clone.returns.expected.stocks,
                      bonds: clone.returns.expected.bonds,
                    }
                    return clone
                  })
                }
              >
                <FontAwesomeIcon
                  className="mr-2"
                  icon={
                    params.returns.historical.type === 'fixed'
                      ? faCircleSelected
                      : faCircleRegular
                  }
                />{' '}
                Fixed
              </h2>
              {params.returns.historical.type === 'fixed' && (
                <_StocksAndBonds className="ml-[28px] mt-2" />
              )}
            </div>
          </div>
          <div
            className="params-card mt-10 flex justify-between items-center"
            style={{padding: paddingCSS(props.sizing.cardPadding)}}
          >
            <h2 className="font-bold text-lg"> Show All Years</h2>
            <ToggleSwitch
              className=""
              enabled={params.display.alwaysShowAllYears}
              setEnabled={x =>
                setParams(p => {
                  const clone = _.cloneDeep(p)
                  clone.display.alwaysShowAllYears = x
                  return clone
                })
              }
            />
          </div>
        </div>
      </ParamsInputBody>
    )
  }
)

const _StocksAndBonds = React.memo(({className = ''}: {className?: string}) => {
  const {params, setParams} = useSimulation()
  assert(params.returns.historical.type === 'fixed')
  const {stocks, bonds} = params.returns.historical
  const [stocksStr, setStocksStr] = useState((stocks * 100).toFixed(1))
  const [bondsStr, setBondsStr] = useState((bonds * 100).toFixed(1))
  useEffect(() => {
    setStocksStr((stocks * 100).toFixed(1))
  }, [stocks])
  useEffect(() => {
    setBondsStr((bonds * 100).toFixed(1))
  }, [bonds])
  const handleStockAmount = (x: number) => {
    if (isNaN(x)) return
    setParams(p => {
      const clone = _.cloneDeep(p)
      assert(clone.returns.historical.type === 'fixed')
      clone.returns.historical.stocks = x / 100
      return clone
    })
  }
  const handleBondAmount = (x: number) => {
    if (isNaN(x)) return
    setParams(p => {
      const clone = _.cloneDeep(p)
      assert(clone.returns.historical.type === 'fixed')
      clone.returns.historical.bonds = x / 100
      return clone
    })
  }
  return (
    <div className={`${className}`}>
      <div
        className=" inline-grid  items-stretch gap-x-4 gap-y-2"
        style={{grid: 'auto/auto 80px auto auto'}}
      >
        <h2 className="self-center">Stocks</h2>
        <input
          type="text"
          pattern="[0-9]"
          inputMode="numeric"
          className=" bg-gray-200 rounded-lg py-1.5 px-2 "
          value={stocksStr}
          onChange={x => setStocksStr(x.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter') handleStockAmount(parseFloat(stocksStr))
          }}
          onBlur={e => handleStockAmount(parseFloat(e.target.value))}
        />
        <button
          className={`flex items-center px-2 `}
          onClick={() => handleStockAmount(stocks * 100 + _delta)}
        >
          <FontAwesomeIcon className="text-base" icon={faPlus} />
        </button>
        <button
          className={`flex items-center px-2 `}
          onClick={() => handleStockAmount(stocks * 100 - _delta)}
        >
          <FontAwesomeIcon className="text-base" icon={faMinus} />
        </button>
        <h2 className="self-center">Bonds</h2>
        <input
          type="text"
          pattern="[0-9]"
          inputMode="numeric"
          className=" bg-gray-200 rounded-lg py-1.5 px-2 "
          value={bondsStr}
          onChange={x => setBondsStr(x.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter') handleBondAmount(parseFloat(bondsStr))
          }}
          onBlur={e => handleBondAmount(parseFloat(e.target.value))}
        />
        <button
          className={`flex items-center px-2 `}
          onClick={() => handleBondAmount(bonds * 100 + _delta)}
        >
          <FontAwesomeIcon className="text-base" icon={faPlus} />
        </button>
        <button
          className={`flex items-center px-2 `}
          onClick={() => handleBondAmount(bonds * 100 - _delta)}
        >
          <FontAwesomeIcon className="text-base" icon={faMinus} />
        </button>
      </div>
    </div>
  )
})

const _delta = 0.1
