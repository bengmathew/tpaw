import _ from 'lodash'
import React from 'react'
import {getDefaultParams} from '../../../TPAWSimulator/DefaultParams'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS} from '../../../Utils/Geometry'
import {preciseRange} from '../../../Utils/PreciseRange'
import {useSimulation} from '../../App/WithSimulation'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputExpectedReturns = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()

    const sliderProps = {
      className: '',
      height: 60,
      formatValue: (x: number) => `${(x * 100).toFixed(1)}%`,
      domain: preciseRange(-0.01, 0.1, 0.001, 3).map((value, i) => ({
        value,
        tick:
          i % 10 === 0
            ? ('large' as const)
            : i % 2 === 0
            ? ('small' as const)
            : ('none' as const),
      })),
    }
    return (
      <ParamsInputBody {...props} headingMarginLeft="normal">
        <div
          className="params-card"
          style={{padding: paddingCSS(props.sizing.cardPadding)}}
        >
          <div
            className="grid my-2 items-center"
            style={{grid: 'auto / auto 1fr'}}
          >
            <Contentful.RichText
              body={content['expected-returns'].intro.fields.body}
              p="col-span-2 mb-2 p-base"
            />

            <h2 className=" whitespace-nowrap">Stocks</h2>
            <SliderInput
              {...sliderProps}
              pointers={[
                {value: params.returns.expected.stocks, type: 'normal'},
              ]}
              onChange={([stocks]) => {
                setParams(params => {
                  const p = _.cloneDeep(params)
                  p.returns.expected.stocks = stocks
                  return p
                })
              }}
            />
            <h2 className="whitespace-nowrap">Bonds</h2>
            <SliderInput
              {...sliderProps}
              pointers={[
                {value: params.returns.expected.bonds, type: 'normal'},
              ]}
              onChange={([bonds]) => {
                setParams(params => {
                  const p = _.cloneDeep(params)
                  p.returns.expected.bonds = bonds
                  return p
                })
              }}
            />
          </div>
          <button
            className="mt-4 underline"
            onClick={() =>
              setParams(p => ({
                ...p,
                returns: getDefaultParams().returns,
              }))
            }
          >
            Reset to Default
          </button>
        </div>
      </ParamsInputBody>
    )
  }
)
