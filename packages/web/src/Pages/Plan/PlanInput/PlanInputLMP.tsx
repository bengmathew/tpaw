import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, { useEffect } from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { paddingCSSStyle } from '../../../Utils/Geometry'
import { smartDeltaFn } from '../../../Utils/SmartDeltaFn'
import { useURLUpdater } from '../../../Utils/UseURLUpdater'
import { assert } from '../../../Utils/Utils'
import { useSimulation } from '../../App/WithSimulation'
import { AmountInput } from '../../Common/Inputs/AmountInput'
import { useGetSectionURL, usePlanContent } from '../Plan'
import { planSectionLabel } from './Helpers/PlanSectionLabel'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps
} from './PlanInputBody/PlanInputBody'

export const PlanInputLMP = React.memo((props: PlanInputBodyPassThruProps) => {
  const {params} = useSimulation()
  const summarySectionURL = useGetSectionURL()('summary')
  const urlUpdater = useURLUpdater()
  useEffect(() => {
    if (params.risk.useTPAWPreset) urlUpdater.replace(summarySectionURL)
  }, [params.risk.useTPAWPreset, summarySectionURL, urlUpdater])
  if (params.risk.useTPAWPreset) return <></>
  return (
    <PlanInputBody {...props}>
      <_LMP className="" props={props} />
    </PlanInputBody>
  )
})

const _LMP = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    assert(!params.risk.useTPAWPreset)
    const currLMP = params.risk.tpawAndSPAW.lmp

    const content = usePlanContent()['lmp']
    const handleAmount = (amount: number) => {
      if (amount === currLMP) return
      const p = _.cloneDeep(params)
      assert(!p.risk.useTPAWPreset)
      p.risk.tpawAndSPAW.lmp = amount
      setParams(p)
    }

    // const handleAutoLMP = async () => {
    //   const clone = _.cloneDeep(params)
    //   params.risk.tpawAndSPAW.lmp = 0
    //   const runInWorker = getTPAWRunInWorkerSingleton()
    //   // await runInWorker.runSimulations({canceled:false}, numRuns, clone, )
    // }

    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <div className="">
          <Contentful.RichText
            body={content.intro[params.strategy]}
            p="p-base"
          />
        </div>
        {/* <button className="btn-dark btn-sm mt-4" onClick={handleAutoLMP}>
          Auto LMP
        </button> */}
        <div className={`flex items-stretch gap-x-2 mt-4`}>
          <AmountInput
            className="text-input"
            prefix="$"
            value={currLMP}
            onChange={handleAmount}
            decimals={0}
            modalLabel={`${planSectionLabel('lmp')}`}
          />
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleAmount(lmpDeltaFn.increment(currLMP))}
          >
            <FontAwesomeIcon className="text-base" icon={faPlus} />
          </button>
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleAmount(lmpDeltaFn.decrement(currLMP))}
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
