import {
  getPartialDefaultPlanParams,
  GlidePath,
  partialDefaultDatelessPlanParams,
} from '@tpaw/common'
import React from 'react'
import { paddingCSSStyle } from '../../../../../Utils/Geometry'
import { GlidePathInput } from '../../../../Common/Inputs/GlidePathInput'
import { useSimulationInfo } from '../../../PlanRootHelpers/WithSimulation'
import { AssetAllocationChart } from '../Helpers/AssetAllocationChart'
import { PlanInputBodyPassThruProps } from '../PlanInputBody/PlanInputBody'
import _ from 'lodash'

export const PlanInputRiskSPAWAndSWRStockAllocationCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParamsNormInstant, updatePlanParams } = useSimulationInfo()
    const handleChange = (x: GlidePath) =>
      updatePlanParams('setSPAWAndSWRAllocation2', x)

    const defaultDatelessAllocation =
      partialDefaultDatelessPlanParams.risk.spawAndSWR.allocation

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg">Stock Allocation</h2>
        <GlidePathInput
          className="mt-4 border border-gray-300 p-2 rounded-lg"
          value={planParamsNormInstant.risk.spawAndSWR.allocation}
          onChange={handleChange}
        />
        <h2 className="mt-6">Graph of this asset allocation:</h2>
        <AssetAllocationChart
          className="mt-4 "
          type="asset-allocation-savings-portfolio"
        />
        <button
          className="mt-6 underline"
          onClick={() => {
            handleChange(
              getPartialDefaultPlanParams(
                planParamsNormInstant.datingInfo.isDated
                  ? {
                      isDatedPlan: true,
                      nowAsCalendarMonth: {
                        year: planParamsNormInstant.datingInfo.nowAsCalendarDay
                          .year,
                        month:
                          planParamsNormInstant.datingInfo.nowAsCalendarDay
                            .month,
                      },
                    }
                  : {
                      isDatedPlan: false,
                    },
              ).risk.spawAndSWR.allocation,
            )
          }}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)
