import { faCircle } from '@fortawesome/pro-light-svg-icons'
import { faCircle as faCircleSolid } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { RadioGroup } from '@headlessui/react'
import _ from 'lodash'
import React from 'react'
import { getDefaultParams } from '../../../TPAWSimulator/DefaultParams'
import { Contentful } from '../../../Utils/Contentful'
import { paddingCSSStyle } from '../../../Utils/Geometry'
import { useSimulation } from '../../App/WithSimulation'
import { usePlanContent } from '../Plan'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps
} from './PlanInputBody/PlanInputBody'

export const PlanInputSimulation = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <_SamplingCard className="" props={props} />
      </PlanInputBody>
    )
  }
)

const _SamplingCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent().simulation
    const handleChange = (x: 'monteCarlo' | 'historical') =>
      setParams(params => {
        const clone = _.cloneDeep(params)
        clone.sampling = x
        return clone
      })
    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <div className="mt-2">
          <Contentful.RichText
            body={content.introSampling[params.strategy]}
            p="col-span-2 mb-2 p-base"
          />
        </div>
        <RadioGroup<'div', 'monteCarlo' | 'historical'>
          value={params.sampling}
          onChange={handleChange}
        >
          <div className="mt-4">
            <RadioGroup.Option<'div', 'monteCarlo' | 'historical'>
              value={'monteCarlo'}
              className="cursor-pointer"
            >
              {({checked}) => (
                <>
                  <div className="flex items-start gap-x-2 py-0.5 ">
                    <FontAwesomeIcon
                      className="mt-1"
                      icon={checked ? faCircleSolid : faCircle}
                    />
                    <RadioGroup.Description as="h2" className={`font-medium`}>
                      Monte Carlo sequence
                    </RadioGroup.Description>
                  </div>
                  <Contentful.RichText
                    body={content.introSamplingMonteCarlo[params.strategy]}
                    p="col-span-2 ml-6 p-base"
                  />
                </>
              )}
            </RadioGroup.Option>
            <RadioGroup.Option<'div', 'monteCarlo' | 'historical'>
              value={'historical'}
              className="cursor-pointer mt-4"
            >
              {({checked}) => (
                <>
                  <div className="flex items-center gap-x-2 ">
                    <FontAwesomeIcon
                      icon={checked ? faCircleSolid : faCircle}
                    />
                    <RadioGroup.Description as="h2" className={`font-medium `}>
                      Historical sequence
                    </RadioGroup.Description>
                  </div>
                  <Contentful.RichText
                    body={content.introSamplingHistorical[params.strategy]}
                    p="col-span-2 ml-6 p-base"
                  />
                </>
              )}
            </RadioGroup.Option>
          </div>
        </RadioGroup>
        <button
          className="mt-6 underline"
          onClick={() => handleChange(getDefaultParams().sampling)}
        >
          Reset to Default
        </button>
      </div>
    )
  }
)
