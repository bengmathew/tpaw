import { fGet } from '@tpaw/common'
import _ from 'lodash'
import { ReactNode, use, useEffect, useMemo } from 'react'
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
  const chartDataForPDF = useChartDataForPDF(type)
  const nonPDFContext = useNonPDFContext()

  return useMemo(
    () => ({ ...chartDataForPDF, ...nonPDFContext }),
    [chartDataForPDF, nonPDFContext],
  )
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
    const { planParamsProcessed } = simulationResult.args
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
    // Get keys from planParamsProcessed, not planParamsNorm because some keys
    // might be in the past, and not show up in planParamsProcessed. We don't
    // want to surface those in the UI.

    planParamsProcessed.byMonth.adjustmentsToSpending.extraSpending.essential.byId.forEach(
      ({ id }) => [id, _add(`spending-essential-${id}`)],
    )

    planParamsProcessed.byMonth.adjustmentsToSpending.extraSpending.discretionary.byId.forEach(
      ({ id }) => [id, _add(`spending-discretionary-${id}`)],
    )
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
  const nonPDFContext = useMemo(
    () => ({ planSizing, planTransitionState }),
    [planSizing, planTransitionState],
  )
  return (
    <NonPDFContext.Provider value={nonPDFContext}>
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
