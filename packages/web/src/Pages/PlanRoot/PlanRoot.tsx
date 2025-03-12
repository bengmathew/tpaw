import { noCase } from '@tpaw/common'
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useURLParam } from '../../Utils/UseURLParam'
import { PlanPrintView } from './PlanRootHelpers/PlanPrintView/PlanPrintView'
import { PlanContent } from './PlanRootHelpers/PlanRootGetStaticProps'
import { WithPlanContent } from './PlanRootHelpers/WithPlanContent'
import { SimulationParams } from './PlanRootHelpers/WithSimulation'
import { PlanRootLink } from './PlanRootLink/PlanRootLink'
import { PlanRootLocalMain } from './PlanRootLocalMain/PlanRootLocalMain'
import { PlanRootServer } from './PlanRootServer/PlanRootServer'
import { PlanRootFile } from './PlanRootFile/PlanRootFile'
import { useFirebaseUser } from '../App/WithFirebaseUser'
import { asyncEffect2 } from '../../Utils/AsyncEffect'
import { useGlobalSuspenseFallbackContext } from '../../../pages/_app'
import { useAssertConst } from '../../Utils/UseAssertConst'
import { useSetGlobalError } from '../App/GlobalErrorBoundary'

type t = Extract<SimulationParams['pdfReportInfo'], { isShowing: false }>

export const PlanRoot = React.memo(
  ({
    planContent,
    src,
  }: {
    planContent: PlanContent
    src:
      | { type: 'serverMain' }
      | { type: 'localMain' }
      | { type: 'serverAlt'; slug: string }
      | { type: 'link' }
      | { type: 'file' }
  }) => {
    const { setGlobalError } = useSetGlobalError()
    const { setGlobalSuspend } = useGlobalSuspenseFallbackContext()

    const firebaseUser = useFirebaseUser()
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
        <WithPlanContent planContent={planContent}>
          {src.type === 'serverMain' ? (
            <PlanRootServer
              key="main"
              src={src}
              pdfReportInfo={pdfReportInfo}
            />
          ) : src.type === 'serverAlt' ? (
            <PlanRootServer
              key={`alt-${src.slug}`}
              src={src}
              pdfReportInfo={pdfReportInfo}
            />
          ) : src.type === 'localMain' ? (
            <PlanRootLocalMain pdfReportInfo={pdfReportInfo} />
          ) : src.type === 'link' ? (
            <PlanRootLink pdfReportInfo={pdfReportInfo} />
          ) : src.type === 'file' ? (
            <PlanRootFile pdfReportInfo={pdfReportInfo} />
          ) : (
            noCase(src)
          )}
        </WithPlanContent>
      </>
    )
  },
)
