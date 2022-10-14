import {faMinus, faPlus} from '@fortawesome/pro-light-svg-icons'
import {faPen} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {Dispatch, useState} from 'react'
import {TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {Contentful} from '../../../Utils/Contentful'
import {formatCurrency} from '../../../Utils/FormatCurrency'
import {paddingCSSStyle} from '../../../Utils/Geometry'
import {smartDeltaFn} from '../../../Utils/SmartDeltaFn'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput} from '../../Common/Inputs/AmountInput'
import {EditLabeledAmount} from '../../Common/Inputs/EditLabeldAmount'
import {usePlanContent} from '../Plan'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

type _State =
  | {type: 'main'}
  | {type: 'edit'; isAdd: boolean; index: number; hideInMain: boolean}

export const PlanInputLegacy = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const [state, setState] = useState<_State>({type: 'main'})
    return (
      <PlanInputBody {...props}>
        <div className="">
          <_TotalTargetCard className="" props={props} />
          <_NonPortfolioSourcesCard
            className="mt-8"
            props={props}
            state={state}
            setState={setState}
          />

          <_RemainderCard className="mt-8" props={props} />
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
                    entries={params => params.legacy.tpawAndSPAW.external}
                    index={state.index}
                  />
                )
              : undefined,
        }}
      </PlanInputBody>
    )
  }
)

const _TotalTargetCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    const handleAmount = (amount: number) => {
      if (amount === params.legacy.tpawAndSPAW.total) return
      const p = _.cloneDeep(params)
      p.legacy.tpawAndSPAW.total = amount
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
            value={params.legacy.tpawAndSPAW.total}
            onChange={handleAmount}
            decimals={0}
            modalLabel="Total Legacy Target"
          />
          <button
            className={`flex items-center px-2 `}
            onClick={() =>
              handleAmount(increment(params.legacy.tpawAndSPAW.total))
            }
          >
            <FontAwesomeIcon className="text-base" icon={faPlus} />
          </button>
          <button
            className={`flex items-center px-2 `}
            onClick={() =>
              handleAmount(decrement(params.legacy.tpawAndSPAW.total))
            }
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
    props: PlanInputBodyPassThruProps
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
              const index = params.legacy.tpawAndSPAW.external.length
              setParams(params => {
                const clone = _.cloneDeep(params)
                clone.legacy.tpawAndSPAW.external.push({
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
          {params.legacy.tpawAndSPAW.external.map(
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
    entry: TPAWParams['legacy']['tpawAndSPAW']['external'][0]
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
              <h2 className="">
                {entry.nominal ? '(nominal dollars)' : '(real dollars)'}
              </h2>
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
    props: PlanInputBodyPassThruProps
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
          {formatCurrency(paramsProcessed.legacy.tpawAndSPAW.target)}{' '}
          <span className="">real</span>
        </h2>
      </div>
    )
  }
)
