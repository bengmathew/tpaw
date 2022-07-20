import {faMinus, faPlus} from '@fortawesome/pro-light-svg-icons'
import {faCircle as faCircleRegular} from '@fortawesome/pro-regular-svg-icons'
import {faCircle as faCircleSelected} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {Switch} from '@headlessui/react'
import _ from 'lodash'
import React, {useMemo, useState} from 'react'
import {extendTPAWParams} from '../../../TPAWSimulator/TPAWParamsExt'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSSStyle} from '../../../Utils/Geometry'
import {fGet, noCase} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput} from '../../Common/Inputs/AmountInput'
import {ToggleSwitch} from '../../Common/Inputs/ToggleSwitch'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

type _Type = 'none' | 'fixedSpending' | 'separateCeilingAndFloor'

export const ParamsInputSpendingCeilingAndFloor = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    const {params, tpawResult, highlightPercentiles} = useSimulation()

    const {asYFN, withdrawalStartYear} = extendTPAWParams(
      tpawResult.args.params.original
    )
    const withdrawalStartAsYFN = asYFN(withdrawalStartYear)
    const content = usePlanContent()

    const [type, setType] = useState<_Type>(
      params.spendingCeiling === null && params.spendingFloor === null
        ? 'none'
        : params.spendingCeiling === params.spendingFloor
        ? 'fixedSpending'
        : 'separateCeilingAndFloor'
    )

    const firstWithdrawalOfFirstHighlightPercentile =
      tpawResult.savingsPortfolio.withdrawals.total.byPercentileByYearsFromNow[
        tpawResult.args.percentiles.indexOf(highlightPercentiles[0])
      ].data[withdrawalStartAsYFN]

    const defaultFloorAmount = _roundUp(
      firstWithdrawalOfFirstHighlightPercentile,
      10000
    )

    return (
      <ParamsInputBody {...props} headingMarginLeft="reduced">
        <div
          className={`params-card`}
          style={{...paddingCSSStyle(props.sizing.cardPadding)}}
        >
          <div className="">
            <Contentful.RichText
              body={
                content['spending-ceiling-and-floor'].intro[params.strategy]
              }
              p=" p-base"
            />
          </div>
          <_NoCeilingOrFloor className="mt-6" type={type} setType={setType} />

          <_SeparateCeilingAndFloor
            className="mt-6"
            type={type}
            setType={setType}
            defaultFloorAmount={defaultFloorAmount}
          />
          <_FixedSpending
            className="mt-6"
            type={type}
            setType={setType}
            defaultFloorAmount={defaultFloorAmount}
          />
        </div>
      </ParamsInputBody>
    )
  }
)

const _NoCeilingOrFloor = React.memo(
  ({
    className = '',
    type,
    setType,
  }: {
    className?: string
    type: _Type
    setType: (type: _Type) => void
  }) => {
    const {setParams} = useSimulation()
    return (
      <div className={`${className}`}>
        <button
          className={`w-full text-left`}
          onClick={() => {
            setParams(params => {
              const clone = _.cloneDeep(params)
              clone.spendingCeiling = null
              clone.spendingFloor = null
              return clone
            })
            setType('none')
          }}
        >
          <FontAwesomeIcon
            className="mr-2"
            icon={type === 'none' ? faCircleSelected : faCircleRegular}
          />{' '}
          No ceiling or floor
        </button>
      </div>
    )
  }
)

const _SeparateCeilingAndFloor = React.memo(
  ({
    className = '',
    type,
    setType,
    defaultFloorAmount,
  }: {
    className?: string
    type: _Type
    setType: (type: _Type) => void
    defaultFloorAmount: number
  }) => {
    const {params, setParams, tpawResult} = useSimulation()

    const {asYFN, withdrawalStartYear} = extendTPAWParams(
      tpawResult.args.params.original
    )
    const withdrawalStartAsYFN = asYFN(withdrawalStartYear)

    const {minWithdrawal, maxWithdrawal} = useMemo(() => {
      const last = fGet(
        _.last(
          tpawResult.savingsPortfolio.withdrawals.total
            .byPercentileByYearsFromNow
        )
      ).data
      const first = fGet(
        _.first(
          tpawResult.savingsPortfolio.withdrawals.total
            .byPercentileByYearsFromNow
        )
      ).data
      const maxWithdrawal = Math.max(...last.slice(withdrawalStartAsYFN))
      const minWithdrawal = Math.min(...first.slice(withdrawalStartAsYFN))
      return {minWithdrawal, maxWithdrawal}
    }, [tpawResult, withdrawalStartAsYFN])

    const defaultCeilingAmount = _roundUp(
      minWithdrawal + (maxWithdrawal - minWithdrawal) / 2,
      10000
    )

    const [hasCeiling, setHasCeiling] = useState(
      type === 'fixedSpending' ? false : params.spendingCeiling !== null
    )
    const [lastCeilingEntry, setLastCeilingEntry] = useState(
      params.spendingCeiling
    )

    const handleCeilingAmount = (amount: number) => {
      if (amount === params.spendingCeiling) return
      amount = Math.max(params.spendingFloor ?? 0, amount)
      setLastCeilingEntry(amount)
      const p = _.cloneDeep(params)
      p.spendingCeiling = amount
      setParams(p)
    }

    const [hasFloor, setHasFloor] = useState(
      type === 'fixedSpending' ? false : params.spendingFloor !== null
    )
    const [lastFloorEntry, setLastFloorEntry] = useState(params.spendingFloor)

    const handleFloorAmount = (amount: number) => {
      if (amount === params.spendingFloor) return
      amount = Math.min(params.spendingCeiling ?? Number.MAX_VALUE, amount)
      setLastFloorEntry(amount)
      const p = _.cloneDeep(params)
      p.spendingFloor = amount
      setParams(p)
    }

    return (
      <div className={`${className}`}>
        <button
          className={`w-full text-left`}
          onClick={() => {
            setParams(params => {
              const clone = _.cloneDeep(params)
              clone.spendingCeiling = hasCeiling ? lastCeilingEntry : null
              clone.spendingFloor = hasFloor ? lastFloorEntry : null
              return clone
            })

            setType('separateCeilingAndFloor')
          }}
        >
          <FontAwesomeIcon
            className="mr-2"
            icon={
              type === 'separateCeilingAndFloor'
                ? faCircleSelected
                : faCircleRegular
            }
          />{' '}
          Separate ceiling and floor
        </button>
        {type === 'separateCeilingAndFloor' && (
          <div
            className="ml-6 grid gap-x-1 pt-4 items-center"
            style={{grid: '35px 50px/auto auto 1fr'}}
          >
            <Switch.Group>
              <Switch.Label className="">Ceiling</Switch.Label>
              <ToggleSwitch
                className=""
                enabled={params.spendingCeiling !== null}
                setEnabled={enabled => {
                  setHasCeiling(enabled)
                  if (!enabled) {
                    const p = _.cloneDeep(params)
                    p.spendingCeiling = null
                    setParams(p)
                  } else {
                    handleCeilingAmount(
                      lastCeilingEntry ?? defaultCeilingAmount
                    )
                  }
                }}
              />
            </Switch.Group>
            {params.spendingCeiling === null ? (
              <div className=""></div>
            ) : (
              <div className={`flex items-stretch `}>
                <AmountInput
                  className={`ml-2 w-[90px] text-input`}
                  prefix="$"
                  value={params.spendingCeiling}
                  onChange={handleCeilingAmount}
                  decimals={0}
                  modalLabel="Spending Ceiling"
                />
                <button
                  className={`flex items-center pl-4 pr-2 `}
                  onClick={() =>
                    handleCeilingAmount(fGet(params.spendingCeiling) + 5000)
                  }
                >
                  <FontAwesomeIcon className="text-base" icon={faPlus} />
                </button>
                <button
                  className={`flex items-center px-2 `}
                  onClick={() =>
                    handleCeilingAmount(
                      Math.max(0, fGet(params.spendingCeiling) - 5000)
                    )
                  }
                >
                  <FontAwesomeIcon className="text-base" icon={faMinus} />
                </button>
              </div>
            )}
            <Switch.Group>
              <Switch.Label className="">Floor</Switch.Label>
              <ToggleSwitch
                className=""
                enabled={params.spendingFloor !== null}
                setEnabled={enabled => {
                  setHasFloor(enabled)
                  if (!enabled) {
                    const p = _.cloneDeep(params)
                    p.spendingFloor = null
                    setParams(p)
                  } else {
                    handleFloorAmount(lastFloorEntry ?? defaultFloorAmount)
                  }
                }}
              />
            </Switch.Group>{' '}
            {params.spendingFloor === null ? (
              <div className=""></div>
            ) : (
              <div className={`flex items-stretch `}>
                <AmountInput
                  className={`ml-2 w-[90px] text-input`}
                  prefix="$"
                  value={params.spendingFloor}
                  onChange={handleFloorAmount}
                  decimals={0}
                  modalLabel="Spending Floor"
                />
                <button
                  className={`flex items-center  pl-4 pr-2  `}
                  onClick={() =>
                    handleFloorAmount(fGet(params.spendingFloor) + 5000)
                  }
                >
                  <FontAwesomeIcon className="text-base" icon={faPlus} />
                </button>
                <button
                  className={`flex items-center px-2 `}
                  onClick={() =>
                    handleFloorAmount(
                      Math.max(0, fGet(params.spendingFloor) - 5000)
                    )
                  }
                >
                  <FontAwesomeIcon className="text-base" icon={faMinus} />
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }
)

const _FixedSpending = React.memo(
  ({
    className = '',
    type,
    setType,
    defaultFloorAmount,
  }: {
    className?: string
    type: _Type
    setType: (type: _Type) => void
    defaultFloorAmount: number
  }) => {
    const {params, setParams} = useSimulation()
    const [value, setValue] = useState(
      params.spendingCeiling === params.spendingFloor
        ? params.spendingCeiling ?? defaultFloorAmount
        : defaultFloorAmount
    )

    const handleFixedAmount = (amount: number) => {
      setValue(amount)
      const p = _.cloneDeep(params)
      p.spendingCeiling = amount
      p.spendingFloor = amount
      setParams(p)
    }
    return (
      <div className={`${className}`}>
        <button
          className={`w-full text-left`}
          onClick={() => {
            handleFixedAmount(value ?? defaultFloorAmount)
            setType('fixedSpending')
          }}
        >
          <FontAwesomeIcon
            className="mr-2"
            icon={type === 'fixedSpending' ? faCircleSelected : faCircleRegular}
          />{' '}
          Fixed spending <span className="">(ceiling = floor)</span>
        </button>
        {type === 'fixedSpending' && (
          <div className={` mt-4 col-span-2 ml-6 flex items-stretch `}>
            <AmountInput
              className={`w-[100px] text-input`}
              prefix="$"
              value={value}
              onChange={handleFixedAmount}
              decimals={0}
              modalLabel="Fixed Spending Amount"
            />
            <button
              className={`flex items-center pl-4 pr-2 `}
              onClick={() =>
                handleFixedAmount(fGet(params.spendingFloor) + 5000)
              }
            >
              <FontAwesomeIcon className="text-base" icon={faPlus} />
            </button>
            <button
              className={`flex items-center px-2 `}
              onClick={() =>
                handleFixedAmount(
                  Math.max(0, fGet(params.spendingFloor) - 5000)
                )
              }
            >
              <FontAwesomeIcon className="text-base" icon={faMinus} />
            </button>
          </div>
        )}
      </div>
    )
  }
)

const _roundUp = (x: number, step: number) => {
  const xPlusStep = x + step
  return xPlusStep - (xPlusStep % step)
}
