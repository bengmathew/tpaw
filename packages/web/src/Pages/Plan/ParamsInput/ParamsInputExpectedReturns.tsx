import _ from 'lodash'
import React from 'react'
import {getDefaultParams} from '../../../TPAWSimulator/DefaultParams'
import {Contentful} from '../../../Utils/Contentful'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {paddingCSS} from '../../../Utils/Geometry'
import {preciseRange} from '../../../Utils/PreciseRange'
import {useSimulation} from '../../App/WithSimulation'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

const suggested = getDefaultParams().returns.expected
const PRESETS = {
  suggested: {...suggested},
  oneOverCAPE: {
    stocks: 0.033,
    bonds: suggested.bonds,
  },
  regressionPrediction: {
    stocks: 0.054,
    bonds: suggested.bonds,
  },
  historical: {stocks: 0.085, bonds: 0.031},
}
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

    const handleChange = (expected: {stocks: number; bonds: number}) => {
      setParams(params => {
        const clone = _.cloneDeep(params)
        clone.returns.expected = expected
        return clone
      })
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
              onChange={([stocks]) =>
                handleChange({stocks, bonds: params.returns.expected.bonds})
              }
            />
            <h2 className="whitespace-nowrap">Bonds</h2>
            <SliderInput
              {...sliderProps}
              pointers={[
                {value: params.returns.expected.bonds, type: 'normal'},
              ]}
              onChange={([bonds]) =>
                handleChange({stocks: params.returns.expected.stocks, bonds})
              }
            />
          </div>
          <h2 className="font-bold mt-8 ">Presets</h2>
          <div
            className=" grid gap-x-4 gap-y-2"
            style={{grid: 'auto/1fr auto auto'}}
          >
            <h2 className=""></h2>
            <h2 className="">Stocks</h2>
            <h2 className="">Bonds</h2>
            <button
              className="text-left underline py-1"
              onClick={() => handleChange({...PRESETS.suggested})}
            >
              Suggested
            </button>
            <h2 className="text-right ">
              {formatPercentage(1)(PRESETS.suggested.stocks)}
            </h2>
            <h2 className="text-right">
              {formatPercentage(1)(PRESETS.suggested.bonds)}
            </h2>
            <button
              className="text-left underline py-1"
              onClick={() => handleChange({...PRESETS.oneOverCAPE})}
            >
              1/CAPE for stocks
            </button>
            <h2 className="text-right ">
              {formatPercentage(1)(PRESETS.oneOverCAPE.stocks)}
            </h2>
            <h2 className="text-right">
              {formatPercentage(1)(PRESETS.oneOverCAPE.bonds)}
            </h2>
            <button
              className="text-left underline py-1"
              onClick={() => handleChange({...PRESETS.regressionPrediction})}
            >
              Regression average for stocks
            </button>
            <h2 className="text-right ">
              {formatPercentage(1)(PRESETS.regressionPrediction.stocks)}
            </h2>
            <h2 className="text-right">
              {formatPercentage(1)(PRESETS.regressionPrediction.bonds)}
            </h2>
            <button
              className="text-left underline py-1"
              onClick={() => handleChange({...PRESETS.historical})}
            >
              Historical average
            </button>
            <h2 className="text-right ">
              {formatPercentage(1)(PRESETS.historical.stocks)}
            </h2>
            <h2 className="text-right">
              {formatPercentage(1)(PRESETS.historical.bonds)}
            </h2>
          </div>
        </div>
      </ParamsInputBody>
    )
  }
)
