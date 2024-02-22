import { fGet } from '@tpaw/common'
import _ from 'lodash'
import { ReactNode, useMemo } from 'react'
import { PERCENTILES_STR } from '../../../UseSimulator/Simulator/Simulator'
import { createContext } from '../../../Utils/CreateContext'
import { useSimulationResult } from '../PlanRootHelpers/WithSimulation'
import {
    PlanResultsChartData,
    PlanResultsChartDataForPDF,
    getPlanResultsChartDataForPDF,
} from './PlanResults/PlanResultsChartCard/PlanResultsChart/PlanResultsChartData'
import { PlanResultsChartType } from './PlanResults/PlanResultsChartType'
import { PlanSizing } from './PlanSizing/PlanSizing'
import { PlanTransitionState } from './PlanTransition'
import { PlanColors } from './UsePlanColors'

type _Value = Map<PlanResultsChartType, PlanResultsChartDataForPDF>

const [PDFContext, usePDFContext] = createContext<_Value>('ChartDataForPDF')
const [NonPDFContext, useNonPDFContext] = createContext<{
  planSizing: PlanSizing
  planTransitionState: PlanTransitionState
}>('ChartDataForNonPDF')

export const useChartDataForPDF = (
  type: PlanResultsChartType,
): PlanResultsChartDataForPDF => fGet(usePDFContext().get(type))

export const useChartData = (
  type: PlanResultsChartType,
): PlanResultsChartData => {
  return { ...useChartDataForPDF(type), ...useNonPDFContext() }
}

export const WithPlanResultsChartDataForPDF = ({
  children,
  planColors,
  layout,
  alwaysShowAllMonths,
}: {
  children: ReactNode
  layout: 'laptop' | 'desktop' | 'mobile'
  planColors: PlanColors
  alwaysShowAllMonths: boolean
}) => {
  const simulationResult = useSimulationResult()
  const value = useMemo(() => {
    const { planParams } = simulationResult.args
    const result = new Map<PlanResultsChartType, PlanResultsChartDataForPDF>()
    const _add = (type: PlanResultsChartType) => {
      result.set(
        type,
        getPlanResultsChartDataForPDF(
          type,
          simulationResult,
          layout,
          planColors,
          alwaysShowAllMonths,
        ),
      )
    }

    _add('spending-total')
    PERCENTILES_STR.forEach((percentile) =>
      _add(`spending-total-funding-sources-${percentile}`),
    )
    _add('spending-general')
    _.values(planParams.adjustmentsToSpending.extraSpending.essential).forEach(
      (x) => [x.id, _add(`spending-essential-${x.id}`)],
    )
    _.values(
      planParams.adjustmentsToSpending.extraSpending.discretionary,
    ).forEach((x) => [x.id, _add(`spending-discretionary-${x.id}`)])
    _add('portfolio')
    _add('asset-allocation-savings-portfolio')
    _add('asset-allocation-total-portfolio')
    _add('withdrawal')

    return result
  }, [alwaysShowAllMonths, layout, planColors, simulationResult])

  return <PDFContext.Provider value={value}>{children}</PDFContext.Provider>
}

export const WithPlanResultsChartData = ({
  children,
  planColors,
  alwaysShowAllMonths,
  planSizing,
  planTransitionState,
}: {
  children: ReactNode
  planColors: PlanColors
  alwaysShowAllMonths: boolean
  planSizing: PlanSizing
  planTransitionState: PlanTransitionState
}) => {
  return (
    <NonPDFContext.Provider value={{ planSizing, planTransitionState }}>
      <WithPlanResultsChartDataForPDF
        planColors={planColors}
        layout={planSizing.args.layout}
        alwaysShowAllMonths={alwaysShowAllMonths}
      >
        {children}
      </WithPlanResultsChartDataForPDF>
    </NonPDFContext.Provider>
  )
}
