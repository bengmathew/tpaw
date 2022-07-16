import {faMinus, faPlus} from '@fortawesome/pro-light-svg-icons'
import {faPen} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {Dispatch, useState} from 'react'
import {TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {Contentful} from '../../../Utils/Contentful'
import {formatCurrency} from '../../../Utils/FormatCurrency'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {paddingCSS, paddingCSSStyle} from '../../../Utils/Geometry'
import {preciseRange} from '../../../Utils/PreciseRange'
import {smartDeltaFn} from '../../../Utils/SmartDeltaFn'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput} from '../../Common/Inputs/AmountInput'
import {EditLabeledAmount} from '../../Common/Inputs/EditLabeldAmount'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'
import {usePlanContent} from '../Plan'
import {paramsInputLabel} from './Helpers/ParamsInputLabel'
import {ParamsInputStrategyConditionCard} from './Helpers/ParamsInputStrategyConditionCard'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

type _State =
  | {type: 'main'}
  | {type: 'edit'; isAdd: boolean; index: number; hideInMain: boolean}

export const ParamsInputLegacy = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    const [state, setState] = useState<_State>({type: 'main'})
    return (
      <ParamsInputBody {...props} headingMarginLeft="normal">
        <div className="">
          <_TotalTargetCard className="" props={props} />
          <_NonPortfolioSourcesCard
            className="mt-8"
            props={props}
            state={state}
            setState={setState}
          />

          <_RemainderCard className="mt-8" props={props} />
          <_StockAllocationCard className="mt-8" props={props} />
        </div>

        {{
          input:
            state.type === 'edit'
              ? transitionOut => (
                  <EditLabeledAmount
                    title={
                      state.isAdd ? 'Add a Legacy Entry' : 'Edit Legacy Entry'
                    }
                    setHideInMain={hideInMain =>
                      setState({...state, hideInMain})
                    }
                    transitionOut={transitionOut}
                    onDone={() => setState({type: 'main'})}
                    entries={params => params.legacy.external}
                    index={state.index}
                  />
                )
              : undefined,
        }}
      </ParamsInputBody>
    )
  }
)

const _TotalTargetCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    const handleAmount = (amount: number) => {
      if (amount === params.legacy.total) return
      const p = _.cloneDeep(params)
      p.legacy.total = amount
      setParams(p)
    }
    const content = usePlanContent()['legacy']
    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg mb-3">Total Legacy Target</h2>
        <Contentful.RichText
          body={content.introAmount[params.strategy]}
          p="p-base"
        />
        <div className={`flex items-center gap-x-2 mt-4`}>
          <AmountInput
            className=" text-input"
            prefix="$"
            value={params.legacy.total}
            onChange={handleAmount}
            decimals={0}
          />
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleAmount(increment(params.legacy.total))}
          >
            <FontAwesomeIcon className="text-base" icon={faPlus} />
          </button>
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleAmount(decrement(params.legacy.total))}
          >
            <FontAwesomeIcon className="text-base" icon={faMinus} />
          </button>
        </div>
      </div>
    )
  }
)

const _NonPortfolioSourcesCard = React.memo(
  ({
    className = '',
    props,
    state,
    setState,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
    state: _State
    setState: Dispatch<_State>
  }) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()['legacy']

    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg mb-3">Non-portfolio Sources</h2>
        <Contentful.RichText
          body={content.introAssets[params.strategy]}
          p="p-base mb-4"
        />
        <div className="flex justify-start gap-x-4 items-center  my-2 ">
          <button
            className="flex items-center justify-center gap-x-2 py-1 pr-2  "
            onClick={() => {
              const index = params.legacy.external.length
              setParams(params => {
                const clone = _.cloneDeep(params)
                clone.legacy.external.push({
                  label: null,
                  value: 0,
                  nominal: false,
                })
                return clone
              })
              setState({type: 'edit', isAdd: true, hideInMain: true, index})
            }}
          >
            <FontAwesomeIcon className="text-2xl" icon={faPlus} />
          </button>
        </div>
        <div className="flex flex-col gap-y-6 mt-4 ">
          {params.legacy.external.map(
            (entry, index) =>
              !(
                state.type === 'edit' &&
                state.hideInMain &&
                state.index === index
              ) && (
                <_Entry
                  key={index}
                  className=""
                  entry={entry}
                  onEdit={() => {
                    setState({
                      type: 'edit',
                      isAdd: false,
                      hideInMain: false,
                      index,
                    })
                  }}
                />
              )
          )}
        </div>
      </div>
    )
  }
)

const _Entry = React.memo(
  ({
    className = '',
    entry,
    onEdit,
  }: {
    className?: string
    entry: TPAWParams['legacy']['external'][0]
    onEdit: () => void
  }) => (
    <div
      className={`${className} flex flex-row justify-between items-stretch rounded-lg `}
    >
      <div className="">
        <h2 className="font-medium">{entry.label ?? '<no label>'}</h2>
        <div className="flex justify-between">
          <div className="flex items-stretch">
            <div className="flex flex-row items-center gap-x-2 mr-2">
              <h2 className="">{formatCurrency(entry.value)}</h2>
              <h2 className="">{entry.nominal ? 'nominal' : 'real'}</h2>
            </div>
          </div>
        </div>
      </div>
      <div className="flex flex-row justify-start items-stretch">
        <button className="px-2 -mr-2" onClick={onEdit}>
          <FontAwesomeIcon className="text-lg" icon={faPen} />
        </button>
      </div>
    </div>
  )
)

const {increment, decrement} = smartDeltaFn([
  {value: 1000000, delta: 100000},
  {value: 2000000, delta: 250000},
])

const _RemainderCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
  }) => {
    const {paramsProcessed} = useSimulation()
    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg mb-3">
          Remainder Funded by Portfolio
        </h2>
        <h2 className="">
          {formatCurrency(paramsProcessed.legacy.target)}{' '}
          <span className="">real</span>
        </h2>
      </div>
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

    return (
      <ParamsInputStrategyConditionCard
        className={`${className} params-card`}
        props={props}
        tpaw
        spaw={false}
        swr={false}
      >
        <h2 className="font-bold text-lg ">Stock Allocation for Legacy</h2>
        {/* {params.strategy === 'SPAW' && (
          <div className="p-base mt-2">
            <span className="bg-gray-300 px-2 rounded-lg ">Note</span>{' '}
            {`You have selected the savings portfolio approach in the "${paramsInputLabel(
              'compare-strategies'
            )}" section. This means that the stock allocation you set here is ignored.`}
          </div>
        )} */}
        <SliderInput
          className={`-mx-3 mt-2`}
          height={60}
          pointers={[
            {
              value: params.targetAllocation.legacyPortfolio.stocks,
              type: 'normal',
            },
          ]}
          onChange={([value]) =>
            setParams(params => {
              const p = _.cloneDeep(params)
              p.targetAllocation.legacyPortfolio.stocks = value
              return p
            })
          }
          formatValue={formatPercentage(0)}
          domain={preciseRange(0, 1, 0.01, 2).map((value, i) => ({
            value: value,
            tick: i % 10 === 0 ? 'large' : i % 2 === 0 ? 'small' : 'none',
          }))}
        />
      </ParamsInputStrategyConditionCard>
    )
  }
)
