import {faMinus, faPlus} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {
  paddingCSS,
  paddingCSSStyle,
  paddingCSSStyleHorz,
} from '../../../Utils/Geometry'
import {preciseRange} from '../../../Utils/PreciseRange'
import {smartDeltaFn} from '../../../Utils/SmartDeltaFn'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput, useAmountInputState} from '../../Common/Inputs/AmountInput'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'
import {Config} from '../../Config'
import {chartPanelLabel} from '../ChartPanel/ChartPanelLabel'
import {usePlanContent} from '../Plan'
import {paramsInputLabel} from './Helpers/ParamsInputLabel'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputRiskAndTimePreference = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    const content = usePlanContent()['risk-and-time-preference']

    return (
      <ParamsInputBody {...props} headingMarginLeft="reduced">
        <div className="">
          <div
            className=""
            style={{
              ...paddingCSSStyleHorz(props.sizing.cardPadding, {scale: 0.5}),
            }}
          >
            <Contentful.RichText
              body={content.intro.fields.body}
              p="mb-2 p-base"
            />
          </div>
          <_StockAllocationCard className="mt-10" props={props} />
          <_SpendingTilt className="mt-10" props={props} />
          {!Config.client.production && (
            <_LMP className="mt-10" props={props} />
          )}
        </div>
      </ParamsInputBody>
    )
  }
)

const _StockAllocationCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()['risk-and-time-preference']

    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg">Stock Allocation</h2>
        <div className="mt-2">
          <Contentful.RichText
            body={content.stockAllocationIntro.fields.body}
            p="p-base"
          />
        </div>
        {params.strategy === 'SPAW' && (
          <div className="p-base mt-2">
            <span className="bg-gray-300 px-2 rounded-lg ">Note</span>{' '}
            {`You have selected the savings portfolio approach in the "${paramsInputLabel(
              'strategy'
            )}" section. This means that the stock allocation you set here is ignored.`}
          </div>
        )}
        <SliderInput
          className={`-mx-3 mt-2
              ${
                params.strategy === 'SPAW'
                  ? 'lighten-2 pointer-events-none'
                  : ''
              }`}
          height={60}
          pointers={[
            {
              value: params.targetAllocation.regularPortfolio.forTPAW.stocks,
              type: 'normal',
            },
          ]}
          onChange={([value]) =>
            setParams(params => {
              const p = _.cloneDeep(params)
              p.targetAllocation.regularPortfolio.forTPAW.stocks = value
              return p
            })
          }
          formatValue={formatPercentage(0)}
          domain={preciseRange(0, 1, 0.01, 2).map((value, i) => ({
            value: value,
            tick: i % 10 === 0 ? 'large' : i % 2 === 0 ? 'small' : 'none',
          }))}
        />
        {params.strategy === 'TPAW' && (
          <div className="p-base mt-2">
            <span className="bg-gray-300 px-2 rounded-lg ">Note</span> The stock
            allocation you set here is on your{' '}
            <span className="">total portfolio</span>. To view the resulting
            asset allocation on your savings portfolio, select{' '}
            {`"${chartPanelLabel(params, 'glide-path', 'short').label.join(
              ' '
            )}"`}{' '}
            from the drop down menu of the graph.
          </div>
        )}
      </div>
    )
  }
)

const _SpendingTilt = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()['risk-and-time-preference']

    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg">Spending Tilt</h2>
        <div className="mt-2">
          <Contentful.RichText
            body={content.spendingTiltIntro.fields.body}
            p="p-base"
          />
        </div>
        <SliderInput
          className="-mx-3 mt-2"
          height={60}
          pointers={[
            {value: params.scheduledWithdrawalGrowthRate, type: 'normal'},
          ]}
          onChange={([value]) =>
            setParams(params => ({
              ...params,
              scheduledWithdrawalGrowthRate: value,
            }))
          }
          formatValue={formatPercentage(1)}
          domain={preciseRange(-0.03, 0.03, 0.001, 3).map((value, i) => ({
            value,
            tick: i % 10 === 0 ? 'large' : i % 1 === 0 ? 'small' : 'none',
          }))}
        />
      </div>
    )
  }
)

const _LMP = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    const valueState = useAmountInputState(params.withdrawals.lmp)

    const content = usePlanContent()['risk-and-time-preference']
    const handleAmount = (amount: number) => {
      if (amount === params.withdrawals.lmp) return
      valueState.setAmountStr(`${amount}`)
      const p = _.cloneDeep(params)
      p.withdrawals.lmp = amount
      setParams(p)
    }

    return (
      <div
        className={`${className} params-card`}
        style={{padding: paddingCSS(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg">LMP</h2>{' '}
        <div className="mt-2">
          <Contentful.RichText body={content.lmpIntro.fields.body} p="p-base" />
        </div>
        <div className={`flex items-stretch gap-x-2 mt-4`}>
          <AmountInput
            className=""
            type="currency"
            state={valueState}
            onAccept={handleAmount}
          />
          <button
            className={`flex items-center px-2 `}
            onClick={() =>
              handleAmount(lmpDeltaFn.increment(params.withdrawals.lmp))
            }
          >
            <FontAwesomeIcon className="text-base" icon={faPlus} />
          </button>
          <button
            className={`flex items-center px-2 `}
            onClick={() =>
              handleAmount(lmpDeltaFn.decrement(params.withdrawals.lmp))
            }
          >
            <FontAwesomeIcon className="text-base" icon={faMinus} />
          </button>
        </div>
      </div>
    )
  }
)

const lmpDeltaFn = smartDeltaFn([
  {value: 1000000, delta: 1000},
  {value: 2000000, delta: 2500},
])
