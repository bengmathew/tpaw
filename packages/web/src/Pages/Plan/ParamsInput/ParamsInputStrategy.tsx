import {faCircle as faCircleRegular} from '@fortawesome/pro-regular-svg-icons'
import {faCircle as faCircleSelected} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {RadioGroup} from '@headlessui/react'
import _ from 'lodash'
import React from 'react'
import {TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS, paddingCSSStyleHorz} from '../../../Utils/Geometry'
import {useSimulation} from '../../App/WithSimulation'
import {GlidePathInput} from '../../Common/Inputs/GlidePathInput'
import {usePlanContent} from '../Plan'
import {AssetAllocationChart} from './Helpers/AssetAllocationChart'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputStrategy = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()['strategy']

    return (
      <ParamsInputBody {...props} headingMarginLeft="reduced">
        <div className="">
          <div
            className="px-2"
            style={{
              ...paddingCSSStyleHorz(props.sizing.cardPadding, {scale: 0.5}),
            }}
          >
            <Contentful.RichText body={content.intro.fields.body} p="p-base" />
          </div>
          <RadioGroup
            value={`${params.strategy}`}
            className="mt-8"
            onChange={(strategy: TPAWParams['strategy']) => {
              setParams(p => {
                const clone = _.cloneDeep(p)
                clone.strategy = strategy
                return clone
              })
            }}
          >
            <RadioGroup.Option
              value="TPAW"
              className=" outline-none params-card"
              style={{padding: paddingCSS(props.sizing.cardPadding)}}
            >
              {({checked}) => (
                <div className="">
                  <RadioGroup.Label as="div" className={`cursor-pointer `}>
                    <h2 className=" font-bold text-lg">
                      <FontAwesomeIcon
                        className="mr-2"
                        icon={checked ? faCircleSelected : faCircleRegular}
                      />{' '}
                      Total portfolio approach
                    </h2>
                    <div className="mt-2">
                      <Contentful.RichText
                        body={content.tpawIntro.fields.body}
                        p="p-base"
                      />
                    </div>
                  </RadioGroup.Label>
                </div>
              )}
            </RadioGroup.Option>

            <RadioGroup.Option
              value="SPAW"
              className="params-card outline-none mt-8"
              style={{padding: paddingCSS(props.sizing.cardPadding)}}
            >
              {({checked}) => (
                <>
                  <RadioGroup.Label
                    as="div"
                    className={`block  cursor-pointer`}
                  >
                    <h2 className=" font-bold text-lg">
                      <FontAwesomeIcon
                        className="mr-2"
                        icon={checked ? faCircleSelected : faCircleRegular}
                      />{' '}
                      Savings portfolio approach
                    </h2>
                    <div className="mt-2">
                      <Contentful.RichText
                        body={content.spawIntro.fields.body}
                        p="p-base"
                      />
                    </div>
                  </RadioGroup.Label>
                  {checked && (
                    <div className="mt-8">
                      <h2 className="font-bold text-l">
                        Asset Allocation on the Savings Portfolio
                      </h2>

                      <GlidePathInput
                        className="mt-4 border border-gray-300 p-2 rounded-lg"
                        value={params.targetAllocation.regularPortfolio.forSPAW}
                        onChange={x =>
                          setParams(p => {
                            const clone = _.cloneDeep(p)
                            clone.targetAllocation.regularPortfolio.forSPAW = x
                            return clone
                          })
                        }
                      />
                      <h2 className="mt-6">Graph of this asset allocation:</h2>
                      <AssetAllocationChart className="mt-4 " />
                    </div>
                  )}
                </>
              )}
            </RadioGroup.Option>
          </RadioGroup>
        </div>
      </ParamsInputBody>
    )
  }
)
