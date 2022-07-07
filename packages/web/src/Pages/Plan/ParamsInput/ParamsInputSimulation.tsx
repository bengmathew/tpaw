import {faCircle} from '@fortawesome/pro-light-svg-icons'
import {faCircle as faCircleSolid} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {RadioGroup} from '@headlessui/react'
import _ from 'lodash'
import React from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS} from '../../../Utils/Geometry'
import {assertFalse} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputSimulation = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    return (
      <ParamsInputBody {...props} headingMarginLeft="normal">
        <>
          <_SamplingCard className="" props={props} />
          {/* <_AdjustCard className="mt-8" props={props} /> */}
        </>
      </ParamsInputBody>
    )
  }
)

const _SamplingCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent().simulation
    return (
      <div
        className={`${className} params-card`}
        style={{padding: paddingCSS(props.sizing.cardPadding)}}
      >
        <div className="mt-2">
          <Contentful.RichText
            body={content.introSampling.fields.body}
            p="col-span-2 mb-2 p-base"
          />
        </div>
        <RadioGroup<'div', 'monteCarlo' | 'historical'>
          value={params.sampling}
          onChange={(x: 'monteCarlo' | 'historical') =>
            setParams(params => {
              const clone = _.cloneDeep(params)
              clone.sampling = x
              return clone
            })
          }
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
                    body={content.introSamplingMonteCarlo.fields.body}
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
                    <RadioGroup.Description
                      as="h2"
                      className={`font-medium `}
                    >
                      Historical sequence
                    </RadioGroup.Description>
                  </div>
                  <Contentful.RichText
                    body={content.introSamplingHistorical.fields.body}
                    p="col-span-2 ml-6 p-base"
                  />
                </>
              )}
            </RadioGroup.Option>
          </div>
        </RadioGroup>
      </div>
    )
  }
)
