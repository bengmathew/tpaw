import {faMinus, faPlus} from '@fortawesome/pro-light-svg-icons'
import {faCircle as faCircleRegular} from '@fortawesome/pro-regular-svg-icons'
import {faCircle as faCircleSelected} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {RadioGroup} from '@headlessui/react'
import _ from 'lodash'
import Link from 'next/link'
import React, {useEffect, useState} from 'react'
import {clearMemoizedRandom} from '../../../TPAWSimulator/Worker/UseTPAWWorker'
import {paddingCSS} from '../../../Utils/Geometry'
import {assert, assertFalse} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput} from '../../Common/Inputs/AmountInput'
import {ToggleSwitch} from '../../Common/Inputs/ToggleSwitch'
import {usePlanContent} from '../Plan'
import { useGetPlanChartURL } from '../PlanChart/UseGetPlanChartURL'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputDev = React.memo((props: PlanInputBodyPassThruProps) => {
  const {params, setParams, paramsProcessed} = useSimulation()

  return (
    <PlanInputBody {...props}>
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
                    stocks: paramsProcessed.returns.expected.stocks,
                    bonds: paramsProcessed.returns.expected.bonds,
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

        <_AdjustCard className="mt-10" props={props} />
        <_Misc className="mt-10" props={props} />
      </div>
    </PlanInputBody>
  )
})

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

const _Misc = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {params, setParams, numRuns, setNumRuns} = useSimulation()
    const getPlanChartURL = useGetPlanChartURL()
    return (
      <div
        className={`${className} params-card`}
        style={{padding: paddingCSS(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg"> Misc</h2>
        <div className=" flex justify-start gap-x-4 items-center mt-4">
          <h2 className=""> Show All Years</h2>
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

        <div className="mt-2 flex gap-x-4 items-center">
          <h2 className="">Number of simulations</h2>
          <AmountInput
            className="text-input"
            value={numRuns}
            onChange={setNumRuns}
            decimals={0}
            modalLabel="Number of Simulations"
          />
        </div>
        <Link href={getPlanChartURL('asset-allocation-total-portfolio')} shallow>
          <a className="block underline pt-4">
            Show Asset Allocation of Total Portfolio Graph
          </a>
        </Link>

        <button
          className="underline pt-4"
          onClick={async () => {
            await clearMemoizedRandom()
            setParams(x => _.cloneDeep(x))
          }}
        >
          Reset random draws
        </button>
      </div>
    )
  }
)

const _AdjustCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent().simulation
    const type = (() => {
      const {historical} = params.returns
      if (historical.type === 'fixed') return 'toExpected' as const
      if (historical.adjust.type === 'to' || historical.adjust.type == 'by') {
        assertFalse()
      }
      return historical.adjust.type
    })()
    return (
      <div
        className={`${className} params-card`}
        style={{padding: paddingCSS(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg">Adjust Historical Returns</h2>
        <RadioGroup<'div', 'none' | 'toExpected'>
          value={type}
          onChange={(type: 'none' | 'toExpected') =>
            setParams(params => {
              const clone = _.cloneDeep(params)
              if (clone.returns.historical.type === 'fixed') {
                return clone
              }
              clone.returns.historical.adjust = {type}
              return clone
            })
          }
        >
          <div className="mt-2">
            <RadioGroup.Option<'div', 'none' | 'toExpected'>
              value={'toExpected'}
              className="flex items-center gap-x-2 py-1.5 cursor-pointer"
            >
              {({checked}) => (
                <>
                  <FontAwesomeIcon
                    icon={checked ? faCircleSelected : faCircleRegular}
                  />
                  <RadioGroup.Description as="h2" className={`py-1`}>
                    Adjust to expected returns
                  </RadioGroup.Description>
                </>
              )}
            </RadioGroup.Option>
            <RadioGroup.Option<'div', 'none' | 'toExpected'>
              value={'none'}
              className="flex items-center gap-x-2 py-1.5 cursor-pointer"
            >
              {({checked}) => (
                <>
                  <FontAwesomeIcon
                    icon={checked ? faCircleSelected : faCircleRegular}
                  />
                  <RadioGroup.Description as="h2" className={``}>
                    Do not adjust
                  </RadioGroup.Description>
                </>
              )}
            </RadioGroup.Option>
          </div>
        </RadioGroup>
      </div>
    )
  }
)
