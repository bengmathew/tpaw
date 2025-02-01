import { faMinus, faPlus } from '@fortawesome/pro-light-svg-icons'
import { faPlus as faPlusThin } from '@fortawesome/pro-thin-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { PlanParams, generateSmallId } from '@tpaw/common'
import React, { useState } from 'react'
import { PlanParamsNormalized } from '../../../../Simulator/NormalizePlanParams/NormalizePlanParams'
import { Contentful } from '../../../../Utils/Contentful'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { paddingCSSStyle } from '../../../../Utils/Geometry'
import { smartDeltaFn } from '../../../../Utils/SmartDeltaFn'
import { trimAndNullify } from '../../../../Utils/TrimAndNullify'
import { AmountInput } from '../../../Common/Inputs/AmountInput'
import { LabelAmountOptMonthRangeInput } from '../../../Common/Inputs/LabelAmountTimedOrUntimedInput/LabeledAmountTimedOrUntimedInput'
import { usePlanContent } from '../../PlanRootHelpers/WithPlanContent'
import {
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../PlanRootHelpers/WithSimulation'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

type _EditState = { isAdd: boolean; entryId: string; hideInMain: boolean }

export const PlanInputLegacy = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const [editState, setEditState] = useState<_EditState | null>(null)

    return (
      <PlanInputBody {...props}>
        <div className="">
          <_TotalTargetCard className="" props={props} />
          <_NonPortfolioSourcesCard
            className="mt-8"
            props={props}
            editState={editState}
            setEditState={setEditState}
          />

          <_RemainderCard className="mt-8" props={props} />
        </div>

        {{
          input: editState
            ? (transitionOut) => (
                <LabelAmountOptMonthRangeInput
                  hasMonthRange={false}
                  addOrEdit={editState.isAdd ? 'add' : 'edit'}
                  title={
                    editState.isAdd ? 'Add a Legacy Entry' : 'Edit Legacy Entry'
                  }
                  labelPlaceholder="E.g. Home Equity"
                  setHideInMain={(hideInMain) =>
                    setEditState({ ...editState, hideInMain })
                  }
                  transitionOut={transitionOut}
                  onDone={() => setEditState(null)}
                  location="legacyExternalSources"
                  entryId={editState.entryId}
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
    const { planParamsNormInstant, updatePlanParams } = useSimulationInfo()
    const handleAmount = (amount: number) =>
      updatePlanParams('setLegacyTotal', amount)

    const content = usePlanContent()['legacy']
    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-3">Total Legacy Target</h2>
        <Contentful.RichText
          body={content.introAmount[planParamsNormInstant.advanced.strategy]}
          p="p-base"
        />
        <div className={`flex items-center gap-x-2 mt-4`}>
          <AmountInput
            className=" text-input"
            prefix="$"
            value={
              planParamsNormInstant.adjustmentsToSpending.tpawAndSPAW.legacy
                .total
            }
            onChange={handleAmount}
            decimals={0}
            modalLabel="Total Legacy Target"
          />
          <button
            className={`flex items-center px-2 `}
            onClick={() =>
              handleAmount(
                increment(
                  planParamsNormInstant.adjustmentsToSpending.tpawAndSPAW.legacy
                    .total,
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
                  planParamsNormInstant.adjustmentsToSpending.tpawAndSPAW.legacy
                    .total,
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
    editState,
    setEditState,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
    editState: _EditState | null
    setEditState: (x: _EditState | null) => void
  }) => {
    const { planParamsNormInstant, updatePlanParams } = useSimulationInfo()
    const content = usePlanContent()['legacy']
    const handleAdd = () => {
      const sortIndex =
        Math.max(
          -1,
          ...planParamsNormInstant.adjustmentsToSpending.tpawAndSPAW.legacy.external.map(
            (x) => x.sortIndex,
          ),
        ) + 1

      const entryId = generateSmallId()
      updatePlanParams('addLabeledAmountUntimed', {
        location: 'legacyExternalSources',
        entryId,
        sortIndex,
      })
      setEditState({ isAdd: true, hideInMain: true, entryId })
    }

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-3">Non-portfolio Sources</h2>
        <Contentful.RichText
          body={content.introAssets[planParamsNormInstant.advanced.strategy]}
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
          {planParamsNormInstant.adjustmentsToSpending.tpawAndSPAW.legacy.external.map(
            (entry) =>
              !(
                editState &&
                editState.hideInMain &&
                editState.entryId === entry.id
              ) && (
                <_Entry
                  key={entry.id}
                  className=""
                  entry={entry}
                  onEdit={() => {
                    setEditState({
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
      className={`${className} block text-start border border-gray-200 rounded-2xl p-3  `}
      onClick={onEdit}
    >
      <div className="">
        <h2 className="font-medium">{entry.label ?? '<no label>'}</h2>
        <div className="flex justify-between">
          <div className="flex items-stretch">
            <div className="flex flex-row items-center gap-x-2 mr-2">
              <h2 className="">{formatCurrency(entry.amount)}</h2>
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
    const { planParamsProcessed } = useSimulationResultInfo().simulationResult

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
            planParamsProcessed.adjustmentsToSpending.tpawAndSpaw.legacy.target,
          )}{' '}
          <span className="">real</span>
        </h2>
      </div>
    )
  },
)

export const PlanInputLegacySummary = React.memo(
  ({ planParamsNorm }: { planParamsNorm: PlanParamsNormalized }) => {
    const { planParamsProcessed } = useSimulationResultInfo().simulationResult
    const { total, external } =
      planParamsNorm.adjustmentsToSpending.tpawAndSPAW.legacy

    return external.length === 0 ? (
      <h2>Target: {formatCurrency(total)} (real dollars)</h2>
    ) : (
      <>
        <div className="grid gap-x-2" style={{ grid: 'auto/1fr auto auto' }}>
          <h2 className="mt-2">Total Target</h2>
          <h2 className="text-right mt-2">{formatCurrency(total)}</h2>
          <h2 className="mt-2">(real dollars)</h2>
          <h2 className=" col-span-3 mt-2">Non-portfolio Sources</h2>
          {external
            .sort((a, b) => a.sortIndex - b.sortIndex)
            .map((x, i) => (
              <React.Fragment key={i}>
                <h2 className="ml-4 mt-1">
                  {trimAndNullify(x.label) ?? '<no label>'}
                </h2>
                <h2 className="mt-1 text-right">{formatCurrency(x.amount)} </h2>
                <h2 className="mt-1">
                  {x.nominal ? '(nominal dollars)' : '(real dollars)'}{' '}
                </h2>
              </React.Fragment>
            ))}
          <h2 className="mt-2">Remaining Target</h2>
          <h2 className="mt-2 text-right">
            {formatCurrency(
              planParamsProcessed.adjustmentsToSpending.tpawAndSpaw.legacy
                .target,
            )}{' '}
          </h2>
          <h2 className="mt-2">(real dollars)</h2>
        </div>
      </>
    )
  },
)
