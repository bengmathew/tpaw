import _ from 'lodash'
import React, {useEffect} from 'react'
import {
  getDefaultPlanParams,
  resolveTPAWRiskPreset,
} from '@tpaw/common'
import {Contentful} from '../../../Utils/Contentful'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {paddingCSSStyle} from '../../../Utils/Geometry'
import {preciseRange} from '../../../Utils/PreciseRange'
import {useURLUpdater} from '../../../Utils/UseURLUpdater'
import {assert} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'
import {useGetSectionURL, usePlanContent} from '../Plan'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputSpendingTilt = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const {params} = useSimulation()
    const summarySectionURL = useGetSectionURL()('summary')
    const urlUpdater = useURLUpdater()
    useEffect(() => {
      if (params.risk.useTPAWPreset) urlUpdater.replace(summarySectionURL)
    }, [params.risk.useTPAWPreset, summarySectionURL, urlUpdater])
    if (params.risk.useTPAWPreset) return <></>
    return (
      <PlanInputBody {...props}>
        <_SpendingTiltCard className="" props={props} />
      </PlanInputBody>
    )
  }
)

const _SpendingTiltCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {params, paramsExt, setParams} = useSimulation()
    assert(!params.risk.useTPAWPreset)
    const content = usePlanContent()['spending-tilt']
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
        <SliderInput
          className="-mx-3 mt-2"
          height={60}
          pointers={[
            {
              value: params.risk.tpawAndSPAW.spendingTilt,
              type: 'normal',
            },
          ]}
          onChange={([value]) =>
            setParams(params => {
              const clone = _.cloneDeep(params)
              assert(!clone.risk.useTPAWPreset)
              clone.risk.tpawAndSPAW.spendingTilt = value
              return clone
            })
          }
          formatValue={formatPercentage(1)}
          domain={preciseRange(-0.03, 0.03, 0.001, 3).map((value, i) => ({
            value,
            tick: i % 10 === 0 ? 'large' : i % 1 === 0 ? 'small' : 'none',
          }))}
        />
        <button
          className="mt-4 underline"
          onClick={() => {
            setParams(params => {
              const clone = _.cloneDeep(params)
              assert(!clone.risk.useTPAWPreset)
              clone.risk.tpawAndSPAW.spendingTilt = resolveTPAWRiskPreset(
                getDefaultPlanParams().risk,
                paramsExt.numYears
              ).tpawAndSPAW.spendingTilt
              return clone
            })
          }}
        >
          Reset to Default
        </button>
      </div>
    )
  }
)
