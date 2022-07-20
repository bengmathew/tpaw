import {
  faCircle as faCircleRegular,
  faMinus,
  faPlus,
} from '@fortawesome/pro-regular-svg-icons'
import {faCircle as faCircleSelected} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {useEffect, useState} from 'react'
import {defaultSWRWithdrawalRate} from '../../../TPAWSimulator/DefaultParams'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSSStyle} from '../../../Utils/Geometry'
import {fGet} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput} from '../../Common/Inputs/AmountInput'
import {smartDeltaFnForAmountInput} from '../../Common/Inputs/SmartDeltaFnForAmountInput'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputWithdrawalRate = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    return (
      <ParamsInputBody {...props} headingMarginLeft="reduced">
        <_WithdrawalCard className="" props={props} />
      </ParamsInputBody>
    )
  }
)

const _WithdrawalCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
  }) => {
    const {params, tpawResult} = useSimulation()
    const content = usePlanContent()['withdrawal']

    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <div className="">
          <Contentful.RichText
            body={content.intro[params.strategy]}
            p=" p-base"
          />
        </div>
        <_Rate className="mt-6" />
        <_Amount className="mt-6" />
      </div>
    )
  }
)

const _Rate = React.memo(({className = ''}: {className?: string}) => {
  const {params, setParams, paramsExt} = useSimulation()
  const {swrWithdrawal} = params
  const {withdrawalsStarted} = paramsExt

  const {numRetirementYears} = paramsExt

  const [lastEntry, setLastEntry] = useState(
    params.swrWithdrawal.type === 'asPercent'
      ? params.swrWithdrawal.percent
      : defaultSWRWithdrawalRate(numRetirementYears)
  )
  useEffect(() => {
    if (params.swrWithdrawal.type === 'asPercent') {
      setLastEntry(params.swrWithdrawal.percent)
    }
  }, [params])

  const rateLabel = withdrawalsStarted
    ? {
        normal: 'Percentage of current portfolio balance',
        title: 'Percentage of current portfolio balance',
      }
    : {
        normal: 'Percentage of savings portfolio at retirement',
        title: 'Percentage of Savings Portfolio at Retirement',
      }

  const handleChange = (percent: number) =>
    setParams(params => {
      const clone = _.cloneDeep(params)
      clone.swrWithdrawal = {type: 'asPercent', percent: _.clamp(percent, 0, 1)}
      return clone
    })
  return (
    <div className={`${className}`}>
      <button
        className={`w-full text-left flex`}
        disabled={swrWithdrawal.type === 'asPercent'}
        onClick={() => handleChange(lastEntry)}
      >
        <FontAwesomeIcon
          className="mr-2 mt-1"
          icon={
            swrWithdrawal.type === 'asPercent'
              ? faCircleSelected
              : faCircleRegular
          }
        />
        <span className="">{rateLabel.normal}</span>
      </button>
      {swrWithdrawal.type === 'asPercent' && (
        <div className="ml-5">
          <div
            className="inline-grid items-stretch mt-4"
            style={{grid: 'auto/auto auto auto'}}
          >
            <AmountInput
              className="text-input w-[60px]"
              value={swrWithdrawal.percent * 100}
              onChange={percent => handleChange(percent / 100)}
              suffix="%"
              decimals={1}
              modalLabel={rateLabel.title}
            />
            <button
              className="ml-2 px-3"
              onClick={() => handleChange(swrWithdrawal.percent + 0.001)}
            >
              <FontAwesomeIcon icon={faPlus} />
            </button>
            <button
              className="px-3"
              onClick={() => handleChange(swrWithdrawal.percent - 0.001)}
            >
              <FontAwesomeIcon icon={faMinus} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
})
const _Amount = React.memo(({className = ''}: {className?: string}) => {
  const {params, setParams, tpawResult, paramsExt} = useSimulation()
  const {swrWithdrawal} = params
  const {asYFN, withdrawalStartYear} = paramsExt

  const [lastEntry, setLastEntry] = useState(
    _.round(
      fGet(
        tpawResult.savingsPortfolio.withdrawals.regular.byPercentileByYearsFromNow.find(
          x => x.percentile === 50
        )
      ).data[asYFN(withdrawalStartYear)],
      -3
    )
  )
  useEffect(() => {
    if (params.swrWithdrawal.type === 'asAmount') {
      setLastEntry(params.swrWithdrawal.amount)
    }
  }, [params])

  const handleChange = (amount: number) =>
    setParams(params => {
      const clone = _.cloneDeep(params)
      clone.swrWithdrawal = {type: 'asAmount', amount}
      return clone
    })
  return (
    <div className={`${className}`}>
      <button
        className={`w-full text-left`}
        disabled={swrWithdrawal.type === 'asAmount'}
        onClick={() => handleChange(lastEntry)}
      >
        <FontAwesomeIcon
          className="mr-2"
          icon={
            params.swrWithdrawal.type === 'asAmount'
              ? faCircleSelected
              : faCircleRegular
          }
        />{' '}
        Amount
      </button>
      {swrWithdrawal.type === 'asAmount' && (
        <div className="ml-5">
          <div
            className="inline-grid items-stretch mt-4"
            style={{grid: 'auto/auto auto auto'}}
          >
            <AmountInput
              className="text-input w-[160px]"
              value={swrWithdrawal.amount}
              onChange={amount => handleChange(amount)}
              prefix="$"
              decimals={0}
              modalLabel="Withdrawal Amount"
            />
            <button
              className="ml-2 px-3"
              onClick={() =>
                handleChange(
                  smartDeltaFnForAmountInput.increment(swrWithdrawal.amount)
                )
              }
            >
              <FontAwesomeIcon icon={faPlus} />
            </button>
            <button
              className="px-3"
              onClick={() =>
                handleChange(
                  smartDeltaFnForAmountInput.decrement(swrWithdrawal.amount)
                )
              }
            >
              <FontAwesomeIcon icon={faMinus} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
})
