import { MarketData, noCase } from '@tpaw/common'
import React, { useEffect, useState } from 'react'
import { useURLParam } from '../../Utils/UseURLParam'
import { PlanPrintView } from './PlanRootHelpers/PlanPrintView/PlanPrintView'
import { PlanContent } from './PlanRootHelpers/PlanRootGetStaticProps'
import { WithMarketData } from './PlanRootHelpers/WithMarketData'
import { WithPlanContent } from './PlanRootHelpers/WithPlanContent'
import { SimulationParams } from './PlanRootHelpers/WithSimulation'
import { WithWASM } from './PlanRootHelpers/WithWASM'
import { PlanRootLink } from './PlanRootLink/PlanRootLink'
import { PlanRootLocalMain } from './PlanRootLocalMain/PlanRootLocalMain'
import { PlanRootServer } from './PlanRootServer/PlanRootServer'

type t = Extract<SimulationParams['pdfReportInfo'], { isShowing: false }>

export const PlanRoot = React.memo(
  ({
    planContent,
    marketData,
    src,
  }: {
    planContent: PlanContent
    marketData: MarketData.Data
    src:
      | { type: 'serverMain' }
      | { type: 'localMain' }
      | { type: 'serverAlt'; slug: string }
      | { type: 'link' }
  }) => {
    const [pdfReportProps, setPDFReportProps] = useState<
      | Parameters<
          Extract<
            SimulationParams['pdfReportInfo'],
            { isShowing: false }
          >['show']
        >[0]
      | null
    >(null)
    const pdfReportInfo: SimulationParams['pdfReportInfo'] = pdfReportProps
      ? {
          isShowing: true,
          onSettings: (settings) =>
            setPDFReportProps({ ...pdfReportProps, settings }),
        }
      : { isShowing: false, show: setPDFReportProps }

    const shouldShowPDF = useURLParam('pdf-report') === 'true'
    useEffect(() => {
      if (!shouldShowPDF) setPDFReportProps(null)
    }, [shouldShowPDF])

    return (
      <>
        {pdfReportProps && <PlanPrintView {...pdfReportProps} />}
        <WithWASM>
          <WithPlanContent planContent={planContent}>
            <WithMarketData marketData={marketData}>
              {src.type === 'serverMain' ? (
                <PlanRootServer
                  key="main"
                  src={src}
                  pdfReportInfo={pdfReportInfo}
                />
              ) : src.type === 'serverAlt' ? (
                <PlanRootServer
                  key={src.slug}
                  src={src}
                  pdfReportInfo={pdfReportInfo}
                />
              ) : src.type === 'localMain' ? (
                <PlanRootLocalMain pdfReportInfo={pdfReportInfo} />
              ) : src.type === 'link' ? (
                <PlanRootLink pdfReportInfo={pdfReportInfo} />
              ) : (
                noCase(src)
              )}
            </WithMarketData>
          </WithPlanContent>
        </WithWASM>
      </>
    )
  },
)
