import _ from 'lodash'
import React from 'react'
import {TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {preciseRange} from '../../../Utils/PreciseRange'
import {StateObj} from '../../../Utils/UseStateObj'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'
import {CardItem} from '../CardItem'

export const ReturnsAndInflationCard = React.memo(
  ({
    params: paramsObj,
  }: {
    params: StateObj<TPAWParams>
  }) => {
    const {value: params, set: setParams} = paramsObj
    const format = formatPercentage(1)
    const subHeading = `Stocks: ${format(
      params.returns.expected.stocks
    )}, Bonds: ${format(params.returns.expected.bonds)}, Inflation: ${format(
      params.inflation
    )}`

    const props = {
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
      <CardItem
        heading="Expected Returns and Inflation"
        subHeading={subHeading}
      >
        <div className="">
          <div
            className="grid my-2 items-center"
            style={{grid: 'auto / auto 1fr'}}
          >
            <p className="col-span-2 mb-2 ">
              Enter the expected real returns for stocks and bonds below. Remember to use real and
              not nominal returns.
            </p>

            <h2 className=" whitespace-nowrap">Stocks</h2>
            <SliderInput
              {...props}
              pointers={[
                {value: params.returns.expected.stocks, type: 'normal'},
              ]}
              onChange={([stocks]) => {
                setParams(params => {
                  const p = _.cloneDeep(params)
                  p.returns.expected.stocks = stocks
                  p.returns.historical.adjust = {
                    type: 'to',
                    ...p.returns.expected,
                  }
                  return p
                })
              }}
            />
            <h2 className="whitespace-nowrap">Bonds</h2>
            <SliderInput
              {...props}
              pointers={[
                {value: params.returns.expected.bonds, type: 'normal'},
              ]}
              onChange={([bonds]) => {
                setParams(params => {
                  const p = _.cloneDeep(params)
                  p.returns.expected.bonds = bonds
                  p.returns.historical.adjust = {
                    type: 'to',
                    ...p.returns.expected,
                  }
                  return p
                })
              }}
            />
            <p className="col-span-2 mt-4 mb-2  ">
              The inflation rate you enter below will be used to convert any
              nominal dollars that you enter in future savings, retirement
              income, and extra withdrawals.
            </p>
            <h2 className="">Inflation</h2>
            <SliderInput
              {...props}
              pointers={[{value: params.inflation, type: 'normal'}]}
              onChange={([inflation]) =>
                setParams(params => ({...params, inflation}))
              }
            />
          </div>
        </div>
      </CardItem>
    )
  }
)
