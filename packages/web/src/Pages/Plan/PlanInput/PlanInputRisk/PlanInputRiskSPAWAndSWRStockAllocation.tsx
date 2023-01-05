import { getDefaultPlanParams } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { paddingCSSStyle } from '../../../../Utils/Geometry'
import { useSimulation } from '../../../App/WithSimulation'
import { GlidePathInput } from '../../../Common/Inputs/GlidePathInput'
import { AssetAllocationChart } from '../Helpers/AssetAllocationChart'
import { PlanInputBodyPassThruProps } from '../PlanInputBody/PlanInputBody'

export const PlanInputRiskSPAWAndSWRStockAllocationCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { params, setParams } = useSimulation()

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg">
          Stock Allocation
        </h2>
        <GlidePathInput
          className="mt-4 border border-gray-300 p-2 rounded-lg"
          value={params.risk.spawAndSWR.allocation}
          onChange={(x) =>
            setParams((params) => {
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
            setParams((params) => {
              const clone = _.cloneDeep(params)
              clone.risk.spawAndSWR.allocation =
                getDefaultPlanParams().risk.spawAndSWR.allocation
              return clone
            })
          }}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)
