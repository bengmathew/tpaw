import {faCircle} from '@fortawesome/pro-light-svg-icons'
import {faCircle as faCircleSolid} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {RadioGroup} from '@headlessui/react'
import _ from 'lodash'
import React, {useEffect} from 'react'
import {getDefaultPlanParams} from '@tpaw/common'
import {riskLevelLabel} from '../../../TPAWSimulator/RiskLevelLabel'
import {TPAWRiskLevel} from '@tpaw/common'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSSStyle} from '../../../Utils/Geometry'
import {useURLUpdater} from '../../../Utils/UseURLUpdater'
import {useSimulation} from '../../App/WithSimulation'
import {useGetSectionURL, usePlanContent} from '../Plan'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputRiskLevel = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const {params} = useSimulation()
    const summarySectionURL = useGetSectionURL()('summary')
    const urlUpdater = useURLUpdater()
    useEffect(() => {
      if (!params.risk.useTPAWPreset) urlUpdater.replace(summarySectionURL)
    }, [params.risk.useTPAWPreset, summarySectionURL, urlUpdater])
    if (!params.risk.useTPAWPreset) return <></>

    return (
      <PlanInputBody {...props}>
        <_RiskLevelCard className="" props={props} />
      </PlanInputBody>
    )
  }
)

const _RiskLevelCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()['risk-level']

    const handleChange = (x: TPAWRiskLevel) => {
      setParams(params => {
        const clone = _.cloneDeep(params)
        clone.risk.tpawPreset = x
        return clone
      })
    }

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
          <RadioGroup<'div', TPAWRiskLevel>
            value={params.risk.tpawPreset}
            onChange={handleChange}
          >
            <div className={`mt-4`}>
              <_Option className="mt-2" riskLevel="riskLevel-1" />
              <_Option className="mt-2" riskLevel="riskLevel-2" />
              <_Option className="mt-2" riskLevel="riskLevel-3" />
              <_Option className="mt-2" riskLevel="riskLevel-4" />
            </div>
          </RadioGroup>
          <p className="p-base mt-6">
            {`If you want to see the settings for these presets, switch to custom mode and select "Copy from a Preset."`}
          </p>
          <button
            className="mt-6 underline"
            onClick={() => handleChange(getDefaultPlanParams().risk.tpawPreset)}
          >
            Reset to Default
          </button>
        </div>
      </div>
    )
  }
)

const _Option = React.memo(
  ({
    riskLevel,
    className = '',
  }: {
    riskLevel: TPAWRiskLevel
    className?: string
  }) => {
    return (
      <RadioGroup.Option<'div', TPAWRiskLevel>
        value={riskLevel}
        className={`${className} cursor-pointer`}
      >
        {({checked}) => (
          <div className="flex items-start gap-x-2 py-0.5 ">
            <FontAwesomeIcon
              className="mt-1"
              icon={checked ? faCircleSolid : faCircle}
            />
            <RadioGroup.Description as="h2" className={``}>
              {riskLevelLabel(riskLevel)}
            </RadioGroup.Description>
          </div>
        )}
      </RadioGroup.Option>
    )
  }
)
