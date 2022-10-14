import _ from 'lodash'
import React, {useEffect} from 'react'
import {
  getDefaultParams,
  resolveTPAWRiskPreset,
} from '../../../TPAWSimulator/DefaultParams'
import {Contentful} from '../../../Utils/Contentful'
import {formatPercentage} from '../../../Utils/FormatPercentage'
import {paddingCSSStyle} from '../../../Utils/Geometry'
import {preciseRange} from '../../../Utils/PreciseRange'
import {useURLUpdater} from '../../../Utils/UseURLUpdater'
import {assert} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {GlidePathInput} from '../../Common/Inputs/GlidePathInput'
import {SliderInput} from '../../Common/Inputs/SliderInput/SliderInput'
import {useGetSectionURL, usePlanContent} from '../Plan'
import {AssetAllocationChart} from './Helpers/AssetAllocationChart'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputStockAllocation = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const {params} = useSimulation()
    const summarySectionURL = useGetSectionURL()('summary')
    const urlUpdater = useURLUpdater()
    useEffect(() => {
      if (params.risk.useTPAWPreset) urlUpdater.replace(summarySectionURL)
    }, [params.risk.useTPAWPreset, summarySectionURL, urlUpdater])
    if (params.risk.useTPAWPreset) return <></>
    const customGuideIntro =
      params.strategy === 'TPAW' ? (
        <span className="">
          <span className="bg-gray-300 rounded-lg px-1 py-0.5 font-semibold">
            Note
          </span>{' '}
          The stock allocation you see here is for the{' '}
          <span className="font-bold italic">total portfolio</span> and is
          different from the stock allocation for the savings portfolio you
          might have encountered in other retirement planners. Here is a guide
          to help you understand how to use this section.
        </span>
      ) : undefined
    return (
      <PlanInputBody {...props} customGuideIntro={customGuideIntro}>
        <>
          {params.strategy === 'TPAW' ? (
            <>
              <_TotalStockAllocationCard className="mt-10" props={props} />
              <_LegacyStockAllocationCard className="mt-10" props={props} />
            </>
          ) : (
            <_SavingsStockAllocationCard className="mt-10" props={props} />
          )}
        </>
      </PlanInputBody>
    )
  }
)

const _TotalStockAllocationCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    assert(!params.risk.useTPAWPreset)

    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg">
          Stock Allocation for Retirement Spending
        </h2>
        <div
          className="mt-2 mb-2 grid items-center gap-x-3"
          style={{grid: 'auto/auto 1fr'}}
        >
          <h2 className="font-">Now</h2>
          <SliderInput
            className={`-mx-3 mt-2 `}
            height={60}
            pointers={[
              {
                value: params.risk.tpaw.allocation.start.stocks,
                type: 'normal',
              },
            ]}
            onChange={([value]) =>
              setParams(params => {
                const clone = _.cloneDeep(params)
                assert(!clone.risk.useTPAWPreset)
                clone.risk.tpaw.allocation.start.stocks = value
                return clone
              })
            }
            formatValue={formatPercentage(0)}
            domain={preciseRange(0, 1, 0.01, 2).map((value, i) => ({
              value: value,
              tick: i % 10 === 0 ? 'large' : i % 2 === 0 ? 'small' : 'none',
            }))}
          />
          <h2 className="font-">Max Age</h2>
          <SliderInput
            className={`-mx-3 mt-2 `}
            height={60}
            pointers={[
              {
                value: params.risk.tpaw.allocation.end.stocks,
                type: 'normal',
              },
            ]}
            onChange={([value]) =>
              setParams(params => {
                const clone = _.cloneDeep(params)
                assert(!clone.risk.useTPAWPreset)
                clone.risk.tpaw.allocation.end.stocks = value
                return clone
              })
            }
            formatValue={formatPercentage(0)}
            domain={preciseRange(0, 1, 0.01, 2).map((value, i) => ({
              value: value,
              tick: i % 10 === 0 ? 'large' : i % 2 === 0 ? 'small' : 'none',
            }))}
          />
        </div>
        <button
          className="mt-4 underline"
          onClick={() => {
            setParams(params => {
              const clone = _.cloneDeep(params)
              assert(!clone.risk.useTPAWPreset)
              clone.risk.tpaw.allocation = resolveTPAWRiskPreset(
                getDefaultParams().risk
              ).tpaw.allocation
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

export const _SavingsStockAllocationCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()

    const content = usePlanContent()['stock-allocation']
    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg">
          Stock Allocation of Savings Portfolio
        </h2>
        <div className="mt-2">
          <Contentful.RichText
            body={content.intro[params.strategy]}
            p="p-base"
          />
        </div>
        <GlidePathInput
          className="mt-4 border border-gray-300 p-2 rounded-lg"
          value={params.risk.spawAndSWR.allocation}
          onChange={x =>
            setParams(params => {
              const clone = _.cloneDeep(params)
              clone.risk.spawAndSWR.allocation = x
              return clone
            })
          }
        />
        <h2 className="mt-6">Graph of this asset allocation:</h2>
        <AssetAllocationChart
          className="mt-4 "
          type="asset-allocation-savings-portfolio"
        />
        <button
          className="mt-6 underline"
          onClick={() => {
            setParams(params => {
              const clone = _.cloneDeep(params)
              assert(!clone.risk.useTPAWPreset)
              clone.risk.spawAndSWR.allocation = resolveTPAWRiskPreset(
                getDefaultParams().risk
              ).spawAndSWR.allocation
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

const _LegacyStockAllocationCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()
    assert(!params.risk.useTPAWPreset)

    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg ">Stock Allocation for Legacy</h2>
        <SliderInput
          className={`-mx-3 mt-2`}
          height={60}
          pointers={[
            {
              value: params.risk.tpaw.allocationForLegacy.stocks,
              type: 'normal',
            },
          ]}
          onChange={([value]) =>
            setParams(params => {
              const clone = _.cloneDeep(params)
              assert(!clone.risk.useTPAWPreset)
              clone.risk.tpaw.allocationForLegacy.stocks = value
              return clone
            })
          }
          formatValue={formatPercentage(0)}
          domain={preciseRange(0, 1, 0.01, 2).map((value, i) => ({
            value: value,
            tick: i % 10 === 0 ? 'large' : i % 2 === 0 ? 'small' : 'none',
          }))}
        />
        <button
          className="mt-4 underline"
          onClick={() => {
            setParams(params => {
              const clone = _.cloneDeep(params)
              assert(!clone.risk.useTPAWPreset)
              clone.risk.tpaw.allocationForLegacy = resolveTPAWRiskPreset(
                getDefaultParams().risk
              ).tpaw.allocationForLegacy
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
