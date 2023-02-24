import { faMinus, faPlus } from '@fortawesome/pro-light-svg-icons'
import { faPlus as faPlusThin } from '@fortawesome/pro-thin-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { PlanParams } from '@tpaw/common'
import _ from 'lodash'
import React, { Dispatch, useState } from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { formatCurrency } from '../../../Utils/FormatCurrency'
import { paddingCSSStyle } from '../../../Utils/Geometry'
import { smartDeltaFn } from '../../../Utils/SmartDeltaFn'
import { useSimulation } from '../../App/WithSimulation'
import { AmountInput } from '../../Common/Inputs/AmountInput'
import { EditValueForMonthRange } from '../../Common/Inputs/EditValueForMonthRange'
import { usePlanContent } from '../Plan'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

type _State =
  | { type: 'main' }
  | { type: 'edit'; isAdd: boolean; entryId: number; hideInMain: boolean }

export const PlanInputLegacy = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const [state, setState] = useState<_State>({ type: 'main' })
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
              ? (transitionOut) => (
                  // <EditLabeledAmount
                  //   title={
                  //     state.isAdd ? 'Add a Legacy Entry' : 'Edit Legacy Entry'
                  //   }
                  //   labelPlaceholder="E.g. Home Equity"
                  //   setHideInMain={(hideInMain) =>
                  //     setState({ ...state, hideInMain })
                  //   }
                  //   transitionOut={transitionOut}
                  //   onDone={() => setState({ type: 'main' })}
                  //   entries={(params) =>
                  //     params.adjustmentsToSpending.tpawAndSPAW.legacy.external
                  //   }
                  //   index={state.index}
                  //   cardPadding={props.sizing.cardPadding}
                  // />
                  <EditValueForMonthRange
                    hasMonthRange={false}
                    mode={state.isAdd ? 'add' : 'edit'}
                    title={
                      state.isAdd ? 'Add a Legacy Entry' : 'Edit Legacy Entry'
                    }
                    labelPlaceholder="E.g. Home Equity"
                    setHideInMain={(hideInMain) =>
                      setState({ ...state, hideInMain })
                    }
                    transitionOut={transitionOut}
                    onDone={() => setState({ type: 'main' })}
                    getEntries={(params) =>
                      params.adjustmentsToSpending.tpawAndSPAW.legacy.external
                    }
                    entryId={state.entryId}
                    cardPadding={props.sizing.cardPadding}
                  />
                )
              : undefined,
        }}
      </PlanInputBody>
    )
  },
)

const _TotalTargetCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { params, setParams } = useSimulation()
    const handleAmount = (amount: number) => {
      if (amount === params.adjustmentsToSpending.tpawAndSPAW.legacy.total)
        return
      const clone = _.cloneDeep(params)
      clone.adjustmentsToSpending.tpawAndSPAW.legacy.total = amount
      setParams(clone)
    }
    const content = usePlanContent()['legacy']
    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-3">Total Legacy Target</h2>
        <Contentful.RichText
          body={content.introAmount[params.advanced.strategy]}
          p="p-base"
        />
        <div className={`flex items-center gap-x-2 mt-4`}>
          <AmountInput
            className=" text-input"
            prefix="$"
            value={params.adjustmentsToSpending.tpawAndSPAW.legacy.total}
            onChange={handleAmount}
            decimals={0}
            modalLabel="Total Legacy Target"
          />
          <button
            className={`flex items-center px-2 `}
            onClick={() =>
              handleAmount(
                increment(
                  params.adjustmentsToSpending.tpawAndSPAW.legacy.total,
                ),
              )
            }
          >
            <FontAwesomeIcon className="text-base" icon={faPlus} />
          </button>
          <button
            className={`flex items-center px-2 `}
            onClick={() =>
              handleAmount(
                decrement(
                  params.adjustmentsToSpending.tpawAndSPAW.legacy.total,
                ),
              )
            }
          >
            <FontAwesomeIcon className="text-base" icon={faMinus} />
          </button>
        </div>
      </div>
    )
  },
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
    const { params, setParams } = useSimulation()
    const content = usePlanContent()['legacy']
    const handleAdd = () => {
      const entryId =
        Math.max(
          -1,
          ...params.adjustmentsToSpending.tpawAndSPAW.legacy.external.map(
            (x) => x.id,
          ),
        ) + 1

      setParams((params) => {
        const clone = _.cloneDeep(params)
        clone.adjustmentsToSpending.tpawAndSPAW.legacy.external.push({
          label: null,
          value: 0,
          nominal: false,
          id: entryId,
        })
        return clone
      })
      setState({ type: 'edit', isAdd: true, hideInMain: true, entryId })
    }

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-3">Non-portfolio Sources</h2>
        <Contentful.RichText
          body={content.introAssets[params.advanced.strategy]}
          p="p-base mb-4"
        />
        <div className="flex justify-start gap-x-4 items-center  my-2 ">
          <button
            className="flex items-center justify-center gap-x-2 py-2 rounded-full border border-gray-200 px-4 "
            onClick={handleAdd}
          >
            <FontAwesomeIcon className="text-3xl" icon={faPlusThin} />
            Add
          </button>
        </div>
        <div className="flex flex-col gap-y-6 mt-4 ">
          {params.adjustmentsToSpending.tpawAndSPAW.legacy.external.map(
            (entry) =>
              !(
                state.type === 'edit' &&
                state.hideInMain &&
                state.entryId === entry.id
              ) && (
                <_Entry
                  key={entry.id}
                  className=""
                  entry={entry}
                  onEdit={() => {
                    setState({
                      type: 'edit',
                      isAdd: false,
                      hideInMain: false,
                      entryId: entry.id,
                    })
                  }}
                />
              ),
          )}
        </div>
      </div>
    )
  },
)

const _Entry = React.memo(
  ({
    className = '',
    entry,
    onEdit,
  }: {
    className?: string
    entry: PlanParams['adjustmentsToSpending']['tpawAndSPAW']['legacy']['external'][0]
    onEdit: () => void
  }) => (
    <button
      // className={`${className} flex flex-row justify-between items-stretch rounded-lg `}
      className={`${className} block text-start border border-gray-200 rounded-2xl p-3  `}
      onClick={onEdit}
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
    </button>
  ),
)

const { increment, decrement } = smartDeltaFn([
  { value: 1000000, delta: 100000 },
  { value: 2000000, delta: 250000 },
])

const _RemainderCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { paramsProcessed } = useSimulation()
    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-3">
          Remainder Funded by Portfolio
        </h2>
        <h2 className="">
          {formatCurrency(
            paramsProcessed.adjustmentsToSpending.tpawAndSPAW.legacy.target,
          )}{' '}
          <span className="">real</span>
        </h2>
      </div>
    )
  },
)
