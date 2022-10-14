import {faTrash} from '@fortawesome/pro-duotone-svg-icons'
import {faCircle} from '@fortawesome/pro-light-svg-icons'
import {faCircle as faCircleSolid} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {RadioGroup} from '@headlessui/react'
import _ from 'lodash'
import React, {useEffect, useState} from 'react'
import {getDefaultParams} from '../../../TPAWSimulator/DefaultParams'
import {riskLevelLabel} from '../../../TPAWSimulator/RiskLevelLabel'
import {TPAWRiskLevel} from '../../../TPAWSimulator/TPAWParams'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSSStyle} from '../../../Utils/Geometry'
import {useURLUpdater} from '../../../Utils/UseURLUpdater'
import {useSimulation} from '../../App/WithSimulation'
import {ConfirmAlert} from '../../Common/Modal/ConfirmAlert'
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
    const [showDelete, setShowDelete] = useState(false)

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
            <div
              className={`mt-4 inline-block ${
                params.risk.customTPAWPreset
                  ? 'border-b border-gray-400 pb-4 mb-4'
                  : ''
              }`}
            >
              <_Option className="mt-2" riskLevel="riskLevel-1" />
              <_Option className="mt-2" riskLevel="riskLevel-2" />
              <_Option className="mt-2" riskLevel="riskLevel-3" />
              <_Option className="mt-2" riskLevel="riskLevel-4" />
            </div>
            {params.risk.customTPAWPreset && (
              <div className="">
                {/* <h2 className="my-4 font-bold border-b border-gray-400 w-full"></h2> */}
                <_Option className="" riskLevel="custom" />
                <p className="p-base mt-2">
                  This is the most recent setting you entered while in the
                  custom mode.{' '}
                  <button
                    className="bg-gray-200 px-2 rounded-full text-base font-font1"
                    onClick={() => setShowDelete(true)}
                  >
                    {/* <FontAwesomeIcon className="text-sm mr-1" icon={faTrash} />{' '} */}
                    Delete
                  </button>
                </p>
              </div>
            )}
          </RadioGroup>
          <button
            className="mt-6 underline"
            onClick={() => handleChange(getDefaultParams().risk.tpawPreset)}
          >
            Reset to Default
          </button>
        </div>
        {showDelete && (
          <ConfirmAlert
            title={null}
            confirmText={'Delete'}
            isWarningButton
            onCancel={() => setShowDelete(false)}
            onConfirm={() => {
              setShowDelete(false)
              setParams(params => {
                const clone = _.cloneDeep(params)
                clone.risk.useTPAWPreset = true
                if (clone.risk.tpawPreset === 'custom')
                  clone.risk.tpawPreset = getDefaultParams().risk.tpawPreset
                clone.risk.customTPAWPreset = null
                return clone
              })
            }}
          >
            Are you sure you want to delete the custom preset?
          </ConfirmAlert>
        )}
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
